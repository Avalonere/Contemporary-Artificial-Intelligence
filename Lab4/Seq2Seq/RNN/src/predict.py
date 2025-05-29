import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from config import Config
from data_loader import DataProcessor
from models.model_factory import ModelFactory
from utils import setup_logging


def predict():
    # 初始化
    config = Config()
    device = config.device
    setup_logging(config.log_dir)

    # 加载数据
    logging.info("Loading test data...")
    processor = DataProcessor(config)
    test_loader = processor.load_test_data()

    # 加载模型
    logging.info(f"Loading {config.model_type} model...")
    model = ModelFactory.create_model(config.model_type, config)
    model_path = Path(config.model_dir) / f"{config.model_type}_best.pt"
    model.load_state_dict(torch.load(model_path, weights_only=False)['model_state'])
    model = model.to(device)
    model.eval()

    # 预测
    logging.info("Starting prediction...")
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            src = batch[0].to(device)
            pred = model.generate(src, max_len=config.max_tgt_len)
            predictions.extend(pred.cpu().tolist())

    # 保存结果
    if not Path(config.output_dir).exists():
        Path(config.output_dir).mkdir(parents=True)

    # 格式化为dataframe并保存
    test_df = pd.read_csv(config.test_file)
    results_df = pd.DataFrame({
        'index': test_df['index'],
        'description': test_df['description'],
        'prediction': [' '.join(map(str, p)) for p in predictions]
    })

    output_path = Path(config.output_dir) / "predictions.csv"
    results_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    predict()
