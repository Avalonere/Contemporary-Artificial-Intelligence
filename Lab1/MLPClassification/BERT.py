# data_utils
import copy
import random
import time

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, is_test=False):
        self.data = []
        self.labels = []
        self.ids = []
        self.tokenizer = tokenizer
        self.is_test = is_test

        if not is_test:
            # Load and organize data by class
            class_data = {i: [] for i in range(10)}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    class_data[item['label']].append(item)

            # Split each class into train and validation
            train_data, val_data = [], []
            for class_label, items in class_data.items():
                train_items, val_items = train_test_split(items, test_size=0.25, random_state=42)
                train_data.extend(train_items)
                val_data.extend(val_items)

            # Shuffle the data
            random.shuffle(train_data)
            random.shuffle(val_data)

            # Store as class attributes
            self.train_data = train_data
            self.val_data = val_data

            # Initialize with train data
            self.current_data = train_data
            for item in self.current_data:
                self.data.append(item['raw'])
                self.labels.append(item['label'])
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    id_, text = line.strip().split(',', 1)
                    self.ids.append(int(id_))
                    self.data.append(text)

    def set_mode(self, mode='train'):
        if not self.is_test:
            if mode == 'train':
                self.current_data = self.train_data
            else:  # validation
                self.current_data = self.val_data

            self.data = []
            self.labels = []
            for item in self.current_data:
                self.data.append(item['raw'])
                self.labels.append(item['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        item = {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(),
                'text': text}

        if not self.is_test:
            item['label'] = torch.tensor(self.labels[idx])

        else:
            item['id'] = self.ids[idx]

        return item


# model
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, num_classes=10, num_hidden_layers=1, dropout_rate=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if num_hidden_layers == 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(768, num_classes)
            )
        elif num_hidden_layers == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        else:  # num_hidden_layers == 2
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


# main
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json


class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_evaluate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    early_stopping = EarlyStopping(patience=3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    best_val_metrics = None
    best_model = None

    epoch_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        total_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training - Loss: {total_loss:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # create a dict to store stats of the epoch
        epoch_stat = {
            'epoch': epoch + 1,
            'train_loss': total_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        epoch_history.append(epoch_stat)
        # Save best model
        if best_val_metrics is None or accuracy > best_val_metrics['accuracy']:
            best_val_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            best_model = model.state_dict().copy()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        if current_lr < 1e-6:
            print("Learning rate too small. Stopping training.")
            break

    return best_val_metrics, best_model, epoch_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='train_data.txt')
    parser.add_argument('--test_file', default='test.txt')
    parser.add_argument('--output_file', default='result')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--lr_patience', type=int, default=2, help='Patience for learning rate reduction')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='Factor by which to reduce learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load datasets
    dataset = TextDataset(args.train_file, tokenizer)
    test_dataset = TextDataset(args.test_file, tokenizer, is_test=True)

    # Create dataloaders

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = copy.deepcopy(train_loader)
    dataset.set_mode('val')
    val_loader = DataLoader(dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Train and evaluate models with different hidden layers
    results = {}
    best_overall_accuracy = 0
    best_num_layers = 0
    all_epoch_his = {}
    for num_hidden_layers in [0]:
        print(f"\nTraining model with {num_hidden_layers} hidden layers")
        # record the time
        start_time = time.time()
        model = BertClassifier(
            num_hidden_layers=num_hidden_layers,
            dropout_rate=args.dropout_rate
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        metrics, model_state, epoch_hist = train_evaluate(
            model, train_loader, val_loader, criterion, optimizer, device, args.epochs
        )
        all_epoch_his[num_hidden_layers] = epoch_hist
        end_time = time.time()
        train_time = end_time - start_time
        results[num_hidden_layers] = metrics
        results[num_hidden_layers]['train_time'] = train_time

        # prediction time
        predictions = []
        ids = []
        model.eval()
        start_pred = time.time()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_ids = batch['id']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs, 1)

                predictions.extend(predicted.cpu().numpy())
                ids.extend(batch_ids.numpy())
        end_pred = time.time()
        pred_time = end_pred - start_pred
        results[num_hidden_layers]['pred_time'] = pred_time

        # Save predictions as file layer_size
        with open(args.output_file + str(num_hidden_layers) + '.txt', 'w') as f:
            f.write("id, pred\n")
            for id_, pred in zip(ids, predictions):
                f.write(f"{id_}, {pred}\n")

        if metrics['accuracy'] > best_overall_accuracy:
            best_overall_accuracy = metrics['accuracy']
            best_num_layers = num_hidden_layers

    print(f"Best accuracy for {best_num_layers} hidden layers: {best_overall_accuracy:.4f}")
    # Save results
    with open('mlp_analysis.json', 'w') as f:
        json.dump(results, f)

    # Save epoch stats
    with open('mlp_epoch_stats.json', 'w') as f:
        json.dump(all_epoch_his, f)


if __name__ == "__main__":
    main()
