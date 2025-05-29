import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from metrics import Metrics


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.best_metrics = None
        self.early_stop = False

    def __call__(self, val_metrics):
        if self.best_metrics is None:
            self.best_metrics = val_metrics
            return False

        if val_metrics > self.best_metrics:
            self.best_metrics = val_metrics
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.patience
        )

        # 评估指标
        self.metrics = Metrics()

        # 学习率调度器
        self.scheduler = None

    def setup_scheduler(self, num_training_steps):
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=num_training_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos',
            final_div_factor=self.config.learning_rate / self.config.min_lr,
            three_phase=True
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            src, tgt = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            logits = self.model(src, tgt)
            loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                  tgt[:, 1:].reshape(-1))

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train(self, train_loader, val_loader):
        """完整训练循环"""
        num_training_steps = len(train_loader) * self.config.max_epochs
        self.setup_scheduler(num_training_steps)

        best_val_metrics = float('inf')
        best_model = None

        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']

            # 记录日志
            current_lr = self.scheduler.get_last_lr()[0]
            logging.info(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            logging.info(f"LR: {current_lr:.2e}")
            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Val Loss: {val_loss:.4f}")
            logging.info(f"BLEU-4 smooth: {val_metrics['bleu4'][0]:.4f}")
            logging.info(f"BLEU-4 non-smooth: {val_metrics['bleu4'][1]:.4f}")
            logging.info(f"Sentence BLEU: {val_metrics['bleu4'][2]:.4f}")
            logging.info(f"METEOR: {val_metrics['meteor']:.4f}")
            logging.info(f"ROUGE-1: {val_metrics['rouge']['avg']['rouge1']:.4f}")
            logging.info(f"ROUGE-2: {val_metrics['rouge']['avg']['rouge2']:.4f}")
            logging.info(f"ROUGE-L: {val_metrics['rouge']['avg']['rougeL']:.4f}")

            avg_metrics = np.mean(
                [val_metrics['bleu4'][0], val_metrics['bleu4'][1], val_metrics['bleu4'][2], val_metrics['meteor'],
                 val_metrics['rouge']['avg']['rouge1'], val_metrics['rouge']['avg']['rouge2'],
                 val_metrics['rouge']['avg']['rougeL']])
            # 保存最佳模型
            if avg_metrics > best_val_metrics:
                best_val_metrics = avg_metrics

                best_model = {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_metrics': val_metrics
                }

            # 早停检查
            if self.early_stopping(avg_metrics):
                logging.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        return best_model

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        refs, hyps = [], []

        with torch.no_grad():
            for batch in val_loader:
                src, tgt = [x.to(self.device) for x in batch]
                logits = self.model(src, tgt)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)),
                                      tgt[:, 1:].reshape(-1))
                total_loss += loss.item()

                # 生成预测序列
                pred = self.model.generate(src)
                refs.extend(tgt.tolist())
                hyps.extend(pred.tolist())

        # 计算指标
        metrics = self.metrics.compute_metrics(refs, hyps)
        metrics['loss'] = total_loss / len(val_loader)

        return metrics
