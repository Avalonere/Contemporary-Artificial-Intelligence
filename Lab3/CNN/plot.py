import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
df = pd.read_csv('./features/ari.csv')

# 设置图形风格和大小
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

# 设置柱状图的位置
x = np.arange(len(df['model']))
width = 0.35

# 绘制柱状图
plt.bar(x - width / 2, df['kmeans_ari'], width, label='K-means', color='skyblue')
plt.bar(x + width / 2, df['dbscan_ari'], width, label='DBSCAN', color='lightcoral')

# 设置图形属性
plt.xlabel('Models')
plt.ylabel('ARI')
plt.title('Clustering Performance Comparison')
plt.xticks(x, df['model'], rotation=45, ha='right')
plt.legend()

# 添加数值标签
for i in range(len(df['model'])):
    plt.text(i - width / 2, df['kmeans_ari'].iloc[i], f'{df["kmeans_ari"].iloc[i]:.3f}',
             ha='center', va='bottom')
    plt.text(i + width / 2, df['dbscan_ari'].iloc[i], f'{df["dbscan_ari"].iloc[i]:.3f}',
             ha='center', va='bottom')

# 调整布局并保存
plt.tight_layout()
plt.savefig('./features/ari_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 使用ggplot风格
plt.style.use('ggplot')

# 创建存储结果的目录
if not os.path.exists('./results'):
    os.makedirs('./results')

# 读取CSV文件
df = pd.read_csv('./results.csv')

# 只保留Scheduler为'plateau'的行
df = df[df['Scheduler'] == 'plateau']

# 设置字体大小
plt.rcParams['font.size'] = 12

# 1. 对比各模型的Test Loss和Test Accuracy
fig, ax1 = plt.subplots(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.35

# 绘制Test Loss
bar1 = ax1.bar(x - width / 2, df['Test Loss'], width, label='Loss')
ax1.set_xlabel('Model')
ax1.set_ylabel('Loss')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])

# # 在柱子上显示具体数值
# for rect in bar1:
#     height = rect.get_height()
#     ax1.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

# 绘制Test Accuracy
ax2 = ax1.twinx()
bar2 = ax2.bar(x + width / 2, df['Test Accuracy'], width, label='Accuracy', color='orange')
ax2.set_ylabel('Accuracy')

# 在柱子上显示具体数值
for rect in bar2:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width() / 2, height, f'{height}', ha='center', va='bottom')

plt.title('Comparison of Test Loss and Test Accuracy')
fig.legend(loc='upper right')

# 保存图像
plt.savefig('./results/test_loss_accuracy.png')
plt.close()

# 2. 对比Test Accuracy, Best Train Acc, Best Val Acc
plt.figure(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.25

bar1 = plt.bar(x - width, df['Test Accuracy'], width, label='Test Accuracy')
bar2 = plt.bar(x, df['Best Train Acc'], width, label='Best Train Acc')
bar3 = plt.bar(x + width, df['Best Val Acc'], width, label='Best Val Acc')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(x, df['Model'])
plt.title('Comparison of Accuracies')
plt.legend()

# # 在柱子上显示具体数值
# for bars in [bar1, bar2, bar3]:
#     for rect in bars:
#         height = rect.get_height()
#         plt.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2%}', ha='center', va='bottom')

plt.savefig('./results/accuracies_comparison.png')
plt.close()

# 3. 对比Training time和Test time
fig, ax1 = plt.subplots(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.35

bar1 = ax1.bar(x - width / 2, df['Training time'], width, label='Training Time')
ax1.set_xlabel('Model')
ax1.set_ylabel('Training Time (s)')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])
#
# for rect in bar1:
#     height = rect.get_height()
#     ax1.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width / 2, df['Test time'], width, label='Test Time', color='orange')
ax2.set_ylabel('Test Time (s)')

# for rect in bar2:
#     height = rect.get_height()
#     ax2.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

plt.title('Comparison of Training and Test Time')
fig.legend(loc='upper right')

plt.savefig('./results/training_test_time.png')
plt.close()

# 4. 对比Epochs和Time/Epoch
fig, ax1 = plt.subplots(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.35

bar1 = ax1.bar(x - width / 2, df['Epochs'], width, label='Epochs')
ax1.set_xlabel('Model')
ax1.set_ylabel('Epochs')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])

# for rect in bar1:
#     height = rect.get_height()
#     ax1.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width / 2, df['Time/Epoch'], width, label='Time/Epoch', color='orange')
ax2.set_ylabel('Time per Epoch (s)')

# for rect in bar2:
#     height = rect.get_height()
#     ax2.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

plt.title('Comparison of Epochs and Time per Epoch')
fig.legend(loc='upper right')

plt.savefig('./results/epochs_time_per_epoch.png')
plt.close()

# 5. 对比Epoches to Get 99% Val Acc和Time to Get 99% Val Acc
fig, ax1 = plt.subplots(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.35

bar1 = ax1.bar(x - width / 2, df['Epoches to Get 99% Val Acc'], width, label='Epochs to 99% Val Acc')
ax1.set_xlabel('Model')
ax1.set_ylabel('Epochs')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])

# for rect in bar1:
#     height = rect.get_height()
#     ax1.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width / 2, df['Time to Get 99% Val Acc'], width, label='Time to 99% Val Acc', color='orange')
ax2.set_ylabel('Time (s)')

# for rect in bar2:
#     height = rect.get_height()
#     ax2.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

plt.title('Time and Epochs to Reach 99% Validation Accuracy')
fig.legend(loc='upper right')

plt.savefig('./results/time_epochs_to_99_val_acc.png')
plt.close()

# 6. 对比Params和Size
fig, ax1 = plt.subplots(figsize=(15, 6))
x = np.arange(len(df['Model']))
width = 0.35

bar1 = ax1.bar(x - width / 2, df['Params'], width, label='Params')
ax1.set_xlabel('Model')
ax1.set_ylabel('Parameters')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Model'])

# for rect in bar1:
#     height = rect.get_height()
#     ax1.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', ha='center', va='bottom')

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width / 2, df['Size'], width, label='Size', color='orange')
ax2.set_ylabel('Size (MB)')

# for rect in bar2:
#     height = rect.get_height()
#     ax2.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

plt.title('Comparison of Model Parameters and Size')
fig.legend(loc='upper right')

plt.savefig('./results/params_size.png')
plt.close()
