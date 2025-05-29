import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_palette("hls")

# 读取数据
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
pred_df = pd.read_csv('runs/bart_9001/pred.csv')

# 基本信息统计
print("训练集大小:", len(train_df))
print("测试集大小:", len(test_df))

# 计算序列长度统计
train_desc_lens = train_df['description'].str.split().apply(len)
train_diag_lens = train_df['diagnosis'].str.split().apply(len)
test_desc_lens = test_df['description'].str.split().apply(len)
pred_diag_lens = pred_df['diagnosis'].str.split().apply(len)

# 可视化序列长度分布
plt.figure(figsize=(12, 6))
plt.subplot(121)
sns.histplot(train_desc_lens, bins=30)
plt.title('Distribution of train description lengths')
plt.xlabel('Sequence Length')

plt.subplot(122)
sns.histplot(train_diag_lens, bins=30)
plt.title('Distribution of train diagnosis lengths')
plt.xlabel('Sequence Length')
plt.tight_layout()
plt.savefig('train_seq_len.png')

# 可视化预测数据长度分布
plt.figure(figsize=(8, 6))
sns.histplot(pred_diag_lens, bins=30)
plt.title('Distribution of prediction diagnosis lengths')
plt.xlabel('Sequence Length')
plt.tight_layout()
plt.savefig('pred_seq_len.png')

# 记录一些关键统计量
print("\nDescription长度统计:")
print(f"最大长度: {train_desc_lens.max()}")
print(f"平均长度: {train_desc_lens.mean():.2f}")
print(f"中位数长度: {train_desc_lens.median()}")

print("\nDiagnosis长度统计:")
print(f"最大长度: {train_diag_lens.max()}")
print(f"平均长度: {train_diag_lens.mean():.2f}")
print(f"中位数长度: {train_diag_lens.median()}")

print("\nPrediction长度统计:")
print(f"最大长度: {pred_diag_lens.max()}")
print(f"平均长度: {pred_diag_lens.mean():.2f}")
print(f"中位数长度: {pred_diag_lens.median()}")
# 计算词表大小
unique_tokens_desc = set()
unique_tokens_diag = set()

for text in train_df['description']:
    unique_tokens_desc.update(text.split())
for text in train_df['diagnosis']:
    unique_tokens_diag.update(text.split())

print(f"\n输入词表大小: {len(unique_tokens_desc)}")
print(f"输出词表大小: {len(unique_tokens_diag)}")
print(max(eval(i) for i in unique_tokens_diag))
