import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define constants
SEEDS = [2333, 4001, 6007, 8009, 9001]
MODELS = ['bart', 't5']

# Prepare data
data = []
for model in MODELS:
    for seed in SEEDS:
        json_path = f'runs/{model}_{seed}/metrics.json'
        if Path(json_path).exists():
            with open(json_path) as f:
                metrics = json.load(f)
                epochs = range(len(metrics['train_loss']))
                data.extend([
                    {'model': model, 'seed': seed, 'epoch': e,
                     'loss': tl, 'type': 'train'}
                    for e, tl in enumerate(metrics['train_loss'])
                ])
                data.extend([
                    {'model': model, 'seed': seed, 'epoch': e,
                     'loss': vl, 'type': 'valid'}
                    for e, vl in enumerate(metrics['valid_loss'])
                ])

df = pd.DataFrame(data)

# Plot
sns.set_theme()
sns.set_palette("hls")
plt.figure(figsize=(10, 6))

# Train loss - solid line
sns.lineplot(data=df, x='epoch', y='loss', hue='model',
            style='type', markers=['o', 's'],
            dashes=False, errorbar='sd')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Save plot
plt.savefig('loss_comparison.png', dpi=300, bbox_inches='tight')

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 定义指标和对应的处理函数
METRICS = ['bleu', 'bleu_no_smooth', 'sentence_bleu', 'meteor', 'rouge1', 'rouge2', 'rougeL']
MODELS = ['t5', 'bart']
SEEDS = [2333, 4001, 6007, 8009, 9001]

def extract_rouge(metrics_data, rouge_type):
    return [epoch[rouge_type] for epoch in metrics_data['valid_rouge']]

def get_metrics_data(model, seed):
    with open(f'runs/{model}_{seed}/metrics.json', 'r') as f:
        data = json.load(f)
        metrics_dict = {
            'bleu': data['valid_bleu'],
            'bleu_no_smooth': data['valid_bleu_no_smooth'],
            'sentence_bleu': data['valid_sentence_bleu'],
            'meteor': data['valid_meteor'],
            'rouge1': extract_rouge(data, 'rouge1'),
            'rouge2': extract_rouge(data, 'rouge2'),
            'rougeL': extract_rouge(data, 'rougeL')
        }
        return metrics_dict

# 准备数据
data = []
for model in MODELS:
    for seed in SEEDS:
        try:
            metrics = get_metrics_data(model, seed)
            epochs = range(len(metrics['bleu']))
            for metric in METRICS:
                for epoch, value in enumerate(metrics[metric]):
                    data.append({
                        'model': model,
                        'seed': seed,
                        'epoch': epoch,
                        'metric': metric,
                        'value': value
                    })
        except FileNotFoundError:
            continue

df = pd.DataFrame(data)


# 绘图
plt.figure(figsize=(12, 8))
colors = sns.husl_palette(n_colors=len(METRICS))
metric_colors = dict(zip(METRICS, colors))

# 使用不同线型区分模型,相同指标用相同颜色
for metric in METRICS:
    metric_data = df[df['metric'] == metric]
    for model in MODELS:
        model_data = metric_data[metric_data['model'] == model]
        linestyle = '-' if model == 't5' else '--'
        sns.lineplot(data=model_data, x='epoch', y='value',
                    label=f'{metric}-{model.upper()}',
                    linestyle=linestyle,
                    color=metric_colors[metric],
                    errorbar='sd')

plt.title('Model Performance Metrics Comparison')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend(bbox_to_anchor=(1.0, 0.75), loc='upper left')
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# 添加在文件末尾

# Plot bleu_no_smooth comparison
plt.figure(figsize=(10, 6))
sns.set_theme()
bleu_data = df[df['metric'] == 'bleu_no_smooth']
sns.lineplot(data=bleu_data, x='epoch', y='value', hue='model',
            errorbar='sd')
plt.title('BLEU Score (No Smoothing) Comparison')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('bleu_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Read and process time data
time_data = []
for model in MODELS:
    for seed in SEEDS:
        try:
            with open(f'runs/{model}_{seed}/time.txt', 'r') as f:
                content = f.read()
                train_time = float(content.split('Train time:')[1].split('s')[0].strip())
                predict_time = float(content.split('Predict time:')[1].split('s')[0].strip())
                time_data.extend([
                    {'model': model, 'seed': seed, 'type': 'Train', 'time': train_time},
                    {'model': model, 'seed': seed, 'type': 'Predict', 'time': predict_time}
                ])
        except FileNotFoundError:
            continue

time_df = pd.DataFrame(time_data)

# Plot time comparison
plt.figure(figsize=(8, 6))
sns.set_theme()
sns.barplot(data=time_df, x='type', y='time', hue='model',
           errorbar='sd', capsize=0.05)
plt.title('Time Comparison between Models')
plt.xlabel('Phase')
plt.ylabel('Time (seconds)')
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('time_comparison.png', dpi=300, bbox_inches='tight')
plt.close()