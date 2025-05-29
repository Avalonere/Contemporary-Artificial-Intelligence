import json

import matplotlib.pyplot as plt


def load_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{filename}'")
        return None
    except json.JSONDecodeError:
        print(f"错误：'{filename}' 不是有效的JSON文件")
        return None


def plot_metrics(data, output_filename):
    plt.figure(figsize=(10, 6))

    # 使用更柔和的颜色
    soft_colors = ['#7986CB',  # 柔和的蓝色
                   '#81C784',  # 柔和的绿色
                   '#FFB74D',  # 柔和的橙色
                   '#BA68C8']  # 柔和的紫色

    hidden_layers = sorted([int(k) for k in data.keys()])
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for metric, color in zip(metrics, soft_colors):
        values = [data[str(hl)][metric] for hl in hidden_layers]
        plt.plot(hidden_layers, values, marker='o', color=color,
                 label=metric.capitalize(), linewidth=2, markersize=8)

    plt.xlabel('Hidden Layers', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Metrics for Different Hidden Layer Configurations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(hidden_layers)

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"指标图片已保存为 '{output_filename}'")
    except Exception as e:
        print(f"保存指标图片时发生错误: {e}")

    plt.close()


def plot_times(data, output_filename):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 使用柔和的颜色
    train_color = '#5C6BC0'  # 柔和的靛蓝色
    pred_color = '#EF9A9A'  # 柔和的粉红色

    hidden_layers = sorted([int(k) for k in data.keys()])

    # 训练时间 - 左轴
    train_times = [data[str(hl)]['train_time'] / 100 for hl in hidden_layers]
    ax1.set_xlabel('Hidden Layers', fontsize=12)
    ax1.set_ylabel('Training Time (100s)', color=train_color, fontsize=12)
    line1 = ax1.plot(hidden_layers, train_times, color=train_color,
                     marker='o', label='Training Time', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=train_color)

    # 预测时间 - 右轴
    ax2 = ax1.twinx()
    pred_times = [data[str(hl)]['pred_time'] for hl in hidden_layers]
    ax2.set_ylabel('Prediction Time (s)', color=pred_color, fontsize=12)
    line2 = ax2.plot(hidden_layers, pred_times, color=pred_color,
                     marker='s', label='Prediction Time', linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=pred_color)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)

    plt.title('Training and Prediction Times for Different Hidden Layer Configurations',
              fontsize=14)
    plt.xticks(hidden_layers)

    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"时间图片已保存为 '{output_filename}'")
    except Exception as e:
        print(f"保存时间图片时发生错误: {e}")

    plt.close()


def main():
    input_file = 'mlp_analysis.json'
    metrics_output = 'mlp_metrics_plot.png'
    times_output = 'mlp_times_plot.png'

    data = load_data(input_file)
    if data is not None:
        plot_metrics(data, metrics_output)
        plot_times(data, times_output)


if __name__ == "__main__":
    main()
