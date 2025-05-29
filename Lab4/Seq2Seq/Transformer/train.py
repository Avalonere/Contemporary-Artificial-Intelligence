import json
import re
from pathlib import Path

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


def post_process_prediction(prediction):
    """后处理预测结果,仅保留数字和空格"""
    # 移除所有非数字字符
    prediction = re.sub(r'[^\d\s]', '', prediction)
    # 规范化空格
    return ' '.join(prediction.split())


def train(model, train_dataset, valid_dataset, tokenizer, config, args):
    model.to(config.device)
    save_dir = Path(f"runs/{args.model}_{args.seed}")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * config.warmup_ratio),
        num_training_steps=num_training_steps
    )

    best_valid_mean = -1
    patience_counter = 0
    metrics_history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_bleu': [],
        'valid_bleu_no_smooth': [],
        'valid_sentence_bleu': [],
        'valid_meteor': [],
        'valid_rouge': [],
        'loss_per_batch': []
    }

    epoch_bar = tqdm(range(args.epochs), desc="Training")
    for epoch in epoch_bar:
        # Training
        model.train()
        total_train_loss = 0
        train_batch_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for batch in train_batch_bar:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_batch_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            metrics_history['loss_per_batch'].append(loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        metrics_history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        total_valid_loss = 0
        all_predictions = []
        all_references = []

        valid_batch_bar = tqdm(valid_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in valid_batch_bar:
                batch = {k: v.to(config.device) for k, v in batch.items()}

                # 计算loss
                outputs = model(**batch)
                loss = outputs.loss
                total_valid_loss += loss.item()

                # 生成预测
                generated_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=config.max_target_length,
                    early_stopping=True,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                    num_beam_groups=config.num_beam_groups,
                    diversity_penalty=config.diversity_penalty
                )

                # 解码预测和参考
                predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
                predictions = [post_process_prediction(pred) for pred in predictions]
                references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=True)

                # 收集结果
                all_predictions.extend([pred.split() for pred in predictions])
                all_references.extend([[ref.split()] for ref in references])

                valid_batch_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算验证指标
        avg_valid_loss = total_valid_loss / len(valid_loader)

        # BLEU
        valid_bleu = corpus_bleu(all_references, all_predictions, smoothing_function=SmoothingFunction().method7)
        valid_bleu_no_smooth = corpus_bleu(all_references, all_predictions,
                                           smoothing_function=SmoothingFunction().method1)

        # sentence_bleu
        valid_sentence_bleu = np.mean([
            sentence_bleu(ref, pred, smoothing_function=SmoothingFunction().method7)
            for ref, pred in zip(all_references, all_predictions)
        ])

        # METEOR
        valid_meteor = np.mean([
            meteor_score([ref[0]], pred)
            for ref, pred in zip(all_references, all_predictions)
        ])

        # 记录rouge1,2,L
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        rouge_scores = [
            scorer.score(" ".join(pred), " ".join(ref[0]))
            for pred, ref in zip(all_predictions, all_references)
        ]
        valid_rouge = {
            "rouge1": np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
            "rouge2": np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
            "rougeL": np.mean([score['rougeL'].fmeasure for score in rouge_scores])
        }

        # save ref and pred
        with open(save_dir / f"valid_epoch_{epoch}.json", "w") as f:
            json.dump({
                "references": all_references,
                "predictions": all_predictions
            }, f, indent=2)

        # 记录指标
        metrics_history['valid_loss'].append(avg_valid_loss)
        metrics_history['valid_bleu'].append(valid_bleu)
        metrics_history['valid_bleu_no_smooth'].append(valid_bleu_no_smooth)
        metrics_history['valid_sentence_bleu'].append(valid_sentence_bleu)
        metrics_history['valid_meteor'].append(valid_meteor)
        metrics_history['valid_rouge'].append(valid_rouge)

        # Early stopping check
        if np.mean([valid_bleu, valid_meteor, valid_rouge['rouge1'], valid_rouge['rouge2'], valid_rouge['rougeL'],
                    valid_bleu_no_smooth, valid_sentence_bleu]) > best_valid_mean:
            best_valid_mean = np.mean([valid_bleu, valid_meteor, valid_rouge['rouge1'], valid_rouge['rouge2'],
                                       valid_rouge['rougeL'], valid_bleu_no_smooth, valid_sentence_bleu])
            patience_counter = 0
            # save the model with args.model
            model.save_pretrained(save_dir / args.model)
            tokenizer.save_pretrained(save_dir / args.model)
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("\nEarly stopping triggered")
            break

        # Update progress bar
        epoch_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}",
            'valid_loss': f"{avg_valid_loss:.4f}",
            'valid_bleu': f"{valid_bleu:.4f}",
            'valid_bleu_no_smooth': f"{valid_bleu_no_smooth:.4f}",
            'valid_sentence_bleu': f"{valid_sentence_bleu:.4f}",
            'valid_meteor': f"{valid_meteor:.4f}",
            'valid_rouge1': f"{valid_rouge['rouge1']:.4f}",
            'valid_rouge2': f"{valid_rouge['rouge2']:.4f}",
            'valid_rougeL': f"{valid_rouge['rougeL']:.4f}",
            'patience': patience_counter,
            'lr': optimizer.param_groups[0]['lr']
        })

    # Save final metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    return model, metrics_history
