import json

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_palette("hls", 2)

# 读取数据
with open('runs/t5_2333/metrics.json', 'r') as f:
    t5_data = json.load(f)

with open('runs/bart_2333/metrics.json', 'r') as f:
    bart_data = json.load(f)

# 获取loss数据
t5_losses = t5_data['loss_per_batch']
bart_losses = bart_data['loss_per_batch']

# 创建batch索引
t5_batches = list(range(1, len(t5_losses) + 1))
bart_batches = list(range(1, len(bart_losses) + 1))

plt.figure(figsize=(12, 6))

# 使用不同的线型和标记
sns.lineplot(x=t5_batches, y=t5_losses, label='T5',
             markersize=6, alpha=0.7,
             linestyle='-', linewidth=2)
sns.lineplot(x=bart_batches, y=bart_losses, label='BART',
             markersize=6, alpha=0.7,
             linestyle='--', linewidth=2)

plt.title('Training Loss Comparison', fontsize=14)
plt.xlabel('Batch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()
