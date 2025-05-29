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


def plot_losses(data, output_filename):
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green']

    for idx, (hidden_layers, stats) in enumerate(data.items()):
        epochs = [item['epoch'] for item in stats]
        train_losses = [item['train_loss'] for item in stats]
        val_losses = [item['val_loss'] for item in stats]

        plt.plot(epochs, train_losses, color=colors[idx], linestyle='-',
                 label=f'Train Loss (Hidden Layers: {hidden_layers})')
        plt.plot(epochs, val_losses, color=colors[idx], linestyle='--',
                 label=f'Val Loss (Hidden Layers: {hidden_layers})')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Different Hidden Layer Configurations')
    plt.legend()
    plt.grid(True)

    try:
        plt.savefig(output_filename)
        print(f"图片已保存为 '{output_filename}'")
    except Exception as e:
        print(f"保存图片时发生错误: {e}")

    plt.close()


def main():
    input_file = 'mlp_epoch_stats.json'
    output_file = 'mlp_loss_plot.png'

    data = load_data(input_file)
    if data is not None:
        plot_losses(data, output_file)


if __name__ == "__main__":
    main()
