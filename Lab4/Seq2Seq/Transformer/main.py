import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import Config
from dataset import Seq2SeqDataset
from predict import predict
from train import train


def main():
    # 设置代理,否则无法下载模型
    os.environ['https_proxy'] = 'http://127.0.0.1:10808'
    os.environ['http_proxy'] = 'http://127.0.0.1:10808'
    os.environ['all_proxy'] = 'socks5://127.0.0.1:10808'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["bart", "t5"], default=Config.model)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.learning_rate)
    parser.add_argument("--epochs", type=int, default=Config.num_epochs)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    # 加载数据
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.1,  # 10%作为验证集
        random_state=args.seed
    )

    # 初始化模型和tokenizer
    model_name = Config.model_name_map[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.model == "t5-base":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 创建dataset
    train_dataset = Seq2SeqDataset(
        train_df,
        tokenizer,
        Config.max_source_length,
        Config.max_target_length
    )

    valid_dataset = Seq2SeqDataset(
        valid_df,
        tokenizer,
        Config.max_source_length,
        Config.max_target_length
    )

    train_start = time.time()
    # 训练
    train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        tokenizer=tokenizer,
        config=Config,
        args=args
    )
    train_end = time.time()
    train_time = train_end - train_start

    # 加载最佳模型进行预测
    print("\nLoading best model for prediction...")
    model_path = Path(f"runs/{args.model}_{args.seed}/{args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(Config.device)

    # 创建测试集
    test_dataset = Seq2SeqDataset(
        test_df,
        tokenizer,
        Config.max_source_length,
        Config.max_target_length
    )

    # 预测
    predict_start = time.time()
    predict(model=model,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            config=Config,
            save_dir=f"runs/{args.model}_{args.seed}",
            batch_size=args.batch_size)
    predict_end = time.time()
    predict_time = predict_end - predict_start

    with open(f"runs/{args.model}_{args.seed}/time.txt", "w") as f:
        f.write(f"Train time: {train_time:.2f} s\n")
        f.write(f"Predict time: {predict_time:.2f} s\n")


if __name__ == "__main__":
    main()
