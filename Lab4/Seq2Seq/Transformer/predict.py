import re
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def post_process_prediction(prediction):
    """后处理预测结果,仅保留数字和空格"""
    prediction = re.sub(r'[^\d\s]', '', prediction)
    return ' '.join(prediction.split())


def predict(model, test_dataset, tokenizer, config, save_dir, batch_size):
    save_dir = Path(save_dir)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = []
    test_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch in test_bar:
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # 生成预测
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=True,
                num_beam_groups=config.num_beam_groups,
                diversity_penalty=config.diversity_penalty,
            )

            # 解码预测和参考
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions = [post_process_prediction(pred) for pred in predictions]

            # 收集结果
            batch_size = len(predictions)
            start_idx = len(results)

            for i in range(batch_size):
                results.append({
                    "index": start_idx + i,
                    "description": test_dataset.data.iloc[start_idx + i]["description"],
                    "diagnosis": predictions[i]
                })

    # 保存预测结果
    pd.DataFrame(results).to_csv(save_dir / "pred.csv", index=False)
