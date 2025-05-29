import glob
import re

import seaborn as sns

sns.set_theme()
sns.set_palette("hls", 7)


def parse_log_file(filepath):
    epochs, train_losses, val_losses = [], [], []
    metrics = {
        'BLEU-4 smooth': [], 'BLEU-4 non-smooth': [], 'Sentence BLEU': [],
        'METEOR': [], 'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []
    }

    current_epoch = None
    with open(filepath, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                match = re.search(r'Epoch (\d+)', line)
                if match:
                    current_epoch = int(match.group(1))
                    epochs.append(current_epoch)
            elif 'Train Loss:' in line:
                train_losses.append(float(line.split('|')[1].split(':')[1].strip()))
            elif 'Val Loss:' in line:
                val_losses.append(float(line.split('|')[1].split(':')[1].strip()))
            else:
                for metric in metrics.keys():
                    if metric in line:
                        metrics[metric].append(float(line.split('|')[1].split(':')[1].strip()))

    return epochs, train_losses, val_losses, metrics


# Read all log files
log_files = glob.glob('./outputs/logs/train.*.log')
models_data = {}

for log_file in log_files:
    model_name = log_file.split('train.')[1].split('.log')[0].upper()
    epochs, train_losses, val_losses, metrics = parse_log_file(log_file)
    models_data[model_name] = {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'metrics': metrics
    }

print(models_data)
# Plot losses
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Define consistent colors and styles
# model_colors = {
#     'RNN': '#2ecc71',
#     'LSTM': '#e74c3c',
#     'GRU': '#3498db'
# }

# Plot losses with seaborn
plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.set_style("whitegrid")

# Prepare data for seaborn
loss_data = []
for model_name, data in models_data.items():
    # Train loss
    loss_data.extend([{
        'Epoch': epoch,
        'Loss': loss,
        'Type': 'Train',
        'Model': model_name
    } for epoch, loss in zip(data['epochs'], data['train_loss'])])

    # Val loss
    loss_data.extend([{
        'Epoch': epoch,
        'Loss': loss,
        'Type': 'Val',
        'Model': model_name
    } for epoch, loss in zip(data['epochs'], data['val_loss'])])

df_loss = pd.DataFrame(loss_data)

# Plot with seaborn
g = sns.lineplot(data=df_loss, x='Epoch', y='Loss',
                 hue='Model', style='Type',
                 markers={'Train': 'o', 'Val': 's'},
                 dashes={'Train': '', 'Val': (2, 2)},

                 markersize=8)

plt.title('Training and Validation Loss')
plt.tight_layout()
plt.savefig('./outputs/losses.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 6))

# Create empty lists to store line objects and labels for legend
lines, labels = [], []
for idx, model_name in enumerate(['RNN', 'LSTM', 'GRU']):
    ax = plt.subplot(1, 3, idx + 1)
    data = models_data[model_name]

    df_metrics = pd.DataFrame(data['metrics'])
    df_metrics.index = data['epochs']

    for metric in df_metrics.columns:
        line = sns.lineplot(data=df_metrics, x=df_metrics.index, y=metric,
                            marker='o', markersize=4, label=metric, ax=ax)

    plt.title(f'{model_name} Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    # Store line objects and labels from first subplot only
    if idx == 0:
        lines = ax.lines
        labels = df_metrics.columns

# Remove individual legends
for ax in plt.gcf().axes:
    ax.get_legend().remove()

# Create shared legend on the right
plt.figlegend(lines, labels,
              bbox_to_anchor=(1.0, 0.5),
              loc='center left',
              title='Metrics')

plt.tight_layout()
plt.savefig('./outputs/model_metrics.png', bbox_inches='tight')
plt.close()

from datetime import datetime

# Calculate running time
time_data = {
    'Model': ['RNN', 'LSTM', 'GRU'],
    'Time (seconds)': [
        (datetime.strptime('2025-01-06 15:24:40', '%Y-%m-%d %H:%M:%S') -
         datetime.strptime('2025-01-06 15:20:25', '%Y-%m-%d %H:%M:%S')).total_seconds(),
        (datetime.strptime('2025-01-06 16:17:55', '%Y-%m-%d %H:%M:%S') -
         datetime.strptime('2025-01-06 15:32:27', '%Y-%m-%d %H:%M:%S')).total_seconds(),
        2193.83
    ]
}

df_time = pd.DataFrame(time_data)

# Create bar plot
plt.figure(figsize=(8, 6))
sns.barplot(data=df_time, x='Model', y='Time (seconds)',
            palette=['#2ecc71', '#e74c3c', '#3498db'])

plt.title('Model Training Time Comparison')
plt.xlabel('Model Type')
plt.ylabel('Training Time (seconds)')

plt.tight_layout()
plt.savefig('./outputs/training_time.png', bbox_inches='tight')
plt.close()
