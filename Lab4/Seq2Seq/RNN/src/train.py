import logging
import time
from pathlib import Path

import torch

from config import Config
from data_loader import DataProcessor
from models.model_factory import ModelFactory
from trainer import Trainer
from utils import set_seed, setup_logging


def main():
    # 初始化
    config = Config()
    set_seed(config.random_seed)

    # 创建输出目录
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(config.log_dir)

    # 加载数据
    logging.info("Loading data...")
    processor = DataProcessor(config)
    train_loader, val_loader = processor.load_and_preprocess()

    # 构建模型
    logging.info(f"Building {config.model_type} model...")
    model = ModelFactory.create_model(config.model_type, config)
    model = model.to(config.device)
    logging.info(f'Model parameters: {model.count_parameters():,}')

    # 训练
    trainer = Trainer(model, config)
    best_val_loss = float('inf')

    # 训练
    logging.info("Starting training...")
    start_train = time.time()
    best_model = trainer.train(train_loader, val_loader)
    end_train = time.time()
    logging.info(f"Training finished in {end_train - start_train:.2f} seconds")

    # 保存最佳模型
    model_path = Path(config.model_dir) / f"{config.model_type}_best.pt"
    torch.save(best_model, model_path)
    logging.info(f"Best model saved to {model_path}")
    # logging.info(f"Best validation metrics: {best_model['val_metrics']}")


if __name__ == "__main__":
    main()
