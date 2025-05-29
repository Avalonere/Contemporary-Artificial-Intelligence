import copy
import json
import random
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


set_seed()


# 数据预处理类
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vectorizer):
        self.texts = vectorizer.transform(texts).toarray()
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.texts[idx]), torch.LongTensor([self.labels[idx]])[0]


# 改进的MLP模型类
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 添加输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[0]))

        # 添加隐藏层
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

        # 添加输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, (layer, bn) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            x = layer(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


# 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    return total_loss / len(train_loader), accuracy_score(true_labels, predictions)


# 评估函数
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return (total_loss / len(test_loader),
            accuracy_score(true_labels, predictions),
            precision_score(true_labels, predictions, average='weighted'),
            recall_score(true_labels, predictions, average='weighted'),
            f1_score(true_labels, predictions, average='weighted'),
            confusion_matrix(true_labels, predictions))


# 早停类
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# ... [保持之前的大部分代码不变，更新 plot_results 函数并添加新的测试函数]

def plot_results(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()

    configs = list(results.keys())

    for idx, metric in enumerate(metrics):
        values = [np.mean(results[config][metric]) for config in configs]

        ax = axes[idx]
        bars = ax.bar(configs, values)

        # 在柱子上方显示具体数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{values[i]:.4f}',
                    ha='center', va='bottom', rotation=0)

        ax.set_title(f'{metric.capitalize()}', fontsize=14)

        # 调整y轴范围以突出差异
        min_val, max_val = min(values), max(values)
        y_range = max_val - min_val
        ax.set_ylim(min_val - y_range * 0.1, max_val + y_range * 0.1)

        # 给y轴添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))

    plt.suptitle('Performance Metrics Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig('tfidf performance_metrics.png')


def find_best_config(results):
    # 使用准确率作为主要指标来选择最佳配置
    config_scores = {
        config: np.mean(results[config]['accuracy'])
        for config in results
    }
    best_config = max(config_scores, key=config_scores.get)
    return eval(best_config)  # 将字符串转换回列表


def predict_test_set(model, vectorizer, test_file, output_file, device):
    print("正在预测测试集...")

    # 读取测试集
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')  # 读取并忽略头行
        for line in f:
            id_, text = line.strip().split(',', 1)
            test_data.append((id_, text))

    # 预处理文本
    preprocessor = TextPreprocessor()
    processed_texts = [preprocessor.preprocess(text) for _, text in test_data]

    # 转换为特征
    X_test = vectorizer.transform(processed_texts).toarray()
    test_dataset = torch.FloatTensor(X_test)

    # 预测
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_dataset), 32):  # 批处理以避免内存问题
            batch = test_dataset[i:i + 32].to(device)
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('id,pred\n')  # 写入头行
        for (id_, _), pred in zip(test_data, predictions):
            f.write(f'{id_},{pred}\n')

    print(f"预测结果已保存到 {output_file}")


def print_detailed_results(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    configs = list(results.keys())

    print("\n详细结果统计：")
    print("配置".ljust(15), end="")
    for metric in metrics:
        print(f"{metric.capitalize().ljust(15)}", end="")
    print()
    print("-" * 60)

    for config in configs:
        print(f"{config.ljust(15)}", end="")
        for metric in metrics:
            values = results[config][metric]
            mean = np.mean(values)
            std = np.std(values)
            print(f"{f'{mean:.4f}±{std:.4f}'.ljust(15)}", end="")
        print()


def main():
    print("正在加载和处理训练数据...")
    # 读取数据并乱序
    with open('train_data.txt', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)

    # 数据预处理
    preprocessor = TextPreprocessor()
    texts = [preprocessor.preprocess(item['raw']) for item in data]
    labels = [item['label'] for item in data]

    print(f"训练集大小: {len(texts)} 样本")
    print(f"标签分布: {pd.Series(labels).value_counts().to_dict()}")

    # 特征提取
    print("正在进行特征提取...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(texts)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 交叉验证
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 不同隐藏层配置
    hidden_layer_configs = [
        [512],
        [512, 512]
    ]

    results = {str(config): {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
               for config in hidden_layer_configs}

    best_models = {}  # 存储每个配置的最佳模型

    for hidden_sizes in hidden_layer_configs:
        print(f"\n开始训练隐藏层配置: {hidden_sizes}")

        config_best_acc = 0
        config_best_model = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
            print(f"正在训练折 {fold + 1}/{k_folds}")

            # 准备数据
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            train_dataset = TextDataset(train_texts, train_labels, vectorizer)
            val_dataset = TextDataset(val_texts, val_labels, vectorizer)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            # 创建模型
            input_size = len(vectorizer.get_feature_names_out())
            num_classes = len(set(labels))
            model = MLPClassifier(input_size, hidden_sizes, num_classes).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=20, steps_per_epoch=len(train_loader))
            early_stopping = EarlyStopping(patience=3)

            best_val_acc = 0
            for epoch in range(20):
                train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler, device)
                val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate_model(
                    model, val_loader, criterion, device)

                print(f"Epoch {epoch + 1}/20:")
                print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
                print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
                print(f"  验证精确率: {val_prec:.4f}, 验证召回率: {val_rec:.4f}, 验证F1: {val_f1:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_metrics = {
                        'accuracy': val_acc,
                        'precision': val_prec,
                        'recall': val_rec,
                        'f1': val_f1,
                    }

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"早停触发，在epoch {epoch + 1}停止训练")
                    break

            # 更新该配置的最佳模型
            if best_val_acc > config_best_acc:
                config_best_acc = best_val_acc
                config_best_model = copy.deepcopy(model)

            # 存储结果
            config_key = str(hidden_sizes)
            for metric, value in best_metrics.items():
                results[config_key][metric].append(value)

        # 保存该配置的最佳模型
        best_models[str(hidden_sizes)] = config_best_model

    # 显示结果
    print_detailed_results(results)
    plot_results(results)

    # 找到最佳配置并使用其对应的模型预测测试集
    best_config = find_best_config(results)
    best_model = best_models[str(best_config)]

    print(f"最佳配置: {best_config}")

    # 预测测试集
    predict_test_set(best_model, vectorizer, 'test.txt', 'result.txt', device)


if __name__ == "__main__":
    main()
