import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from config import Config


class TextDataset(Dataset):
    def __init__(self, descriptions, diagnoses, tokenizer):
        self.descriptions = descriptions
        self.diagnoses = diagnoses
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        desc = [int(x) for x in self.descriptions[idx].strip().split()]
        diag = [int(x) for x in self.diagnoses[idx].strip().split()]
        return torch.tensor(desc), torch.tensor(diag)


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.vocab = None

    def load_and_preprocess(self):
        # 读取数据
        train_df = pd.read_csv(self.config.train_file)

        # 划分训练集和验证集
        train_data, val_data = train_test_split(
            train_df,
            test_size=self.config.validation_split,
            random_state=self.config.random_seed
        )

        # 创建数据集实例
        train_dataset = TextDataset(
            train_data['description'].values,
            train_data['diagnosis'].values,
            self.vocab
        )

        val_dataset = TextDataset(
            val_data['description'].values,
            val_data['diagnosis'].values,
            self.vocab
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return train_loader, val_loader

    def load_test_data(self):
        test_df = pd.read_csv(self.config.test_file)
        test_dataset = TextDataset(
            test_df['description'].values,
            test_df['diagnosis'].values,
            self.vocab
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return test_loader

    @staticmethod
    def collate_fn(batch):
        src_tensors = [x[0] for x in batch]
        tgt_tensors = [x[1] for x in batch]

        # 填充序列
        src_tensors = pad_sequence(src_tensors, batch_first=True)
        tgt_tensors = pad_sequence(tgt_tensors, batch_first=True)

        return src_tensors, tgt_tensors

    def verify_data(self):
        """验证数据处理的正确性"""
        train_loader, val_loader = self.load_and_preprocess()

        # 检查一个batch的数据
        src_batch, tgt_batch = next(iter(train_loader))
        print(f"Source batch shape: {src_batch.shape}")
        print(f"Target batch shape: {tgt_batch.shape}")

        # 验证序列长度
        print(f"Max source length: {src_batch.size(1)}")
        print(f"Max target length: {tgt_batch.size(1)}")

        return True


if __name__ == "__main__":
    config = Config()
    processor = DataProcessor(config)
    processor.verify_data()
