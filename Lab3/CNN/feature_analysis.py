import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.AlexNet import AlexNet
from models.EfficientNet import EfficientNet
from models.GoogLeNet import GoogLeNet
from models.LeNet import LeNet
from models.MNASNet import MNASNet
from models.MobileNetV3Small import MobileNetV3Small
from models.ResNet18 import ResNet18
from models.ShuffleNetV2 import ShuffleNetV2
from models.SqueezeNet import SqueezeNet
from models.VGG11 import VGG11


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


def extract_features(model, data_loader, device):
    features = []
    labels = []
    model.eval()

    with torch.no_grad():
        for images, target in data_loader:
            images = images.to(device)
            feature = model(images)
            features.append(feature.cpu().numpy())
            labels.append(target.numpy())

    return np.vstack(features), np.concatenate(labels)


def plot_features(features, labels, kmeans_clusters, dbscan_clusters, model_name):
    plt.figure(figsize=(18, 6))
    plt.style.use('ggplot')

    # 真实标签可视化
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels,
                          cmap='Set3', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE with True Labels')

    # K-means聚类结果
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(features[:, 0], features[:, 1], c=kmeans_clusters,
                          cmap='Set3', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE with K-means Clusters')

    # DBSCAN聚类结果
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(features[:, 0], features[:, 1], c=dbscan_clusters,
                          cmap='Set3', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE with DBSCAN Clusters')

    plt.tight_layout()
    plt.savefig(f'./features/feature_analysis_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def find_best_dbscan_params(features, true_labels, k_dist=4):
    """
    使用ARI寻找最佳DBSCAN参数
    """
    print("Computing KNN distances...")
    nbrs = NearestNeighbors(n_neighbors=k_dist).fit(features)
    distances, _ = nbrs.kneighbors(features)
    distances = np.sort(distances[:, -1])

    knee_point = np.diff(distances)
    candidate_eps = distances[:-1][np.argmax(knee_point)]

    eps_range = np.linspace(candidate_eps * 0.5, candidate_eps * 1.5, 10)
    min_samples_range = [5, 10, 15, 20, 25, 30]

    best_ari = -1
    best_params = None
    best_labels = None

    print("Searching for best parameters...")
    for eps in tqdm(eps_range, desc='EPS'):
        for min_samples in tqdm(min_samples_range, desc='Min Samples', leave=False):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            pred_labels = dbscan.fit_predict(features)

            # 跳过无效聚类结果
            n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
            if n_clusters <= 1:
                continue

            # 计算ARI得分
            ari = adjusted_rand_score(true_labels, pred_labels)
            if ari > best_ari:
                best_ari = ari
                best_params = (eps, min_samples)
                best_labels = pred_labels

    print(f"Best ARI score: {best_ari:.3f}")
    return best_params, best_labels


def plot_knn_distance(features, model_name, k_dist=4):
    """
    绘制KNN距离图以辅助选择eps
    """
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    nbrs = NearestNeighbors(n_neighbors=k_dist).fit(features)
    distances, _ = nbrs.kneighbors(features)
    distances = np.sort(distances[:, -1])

    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k_dist}-NN Distance')
    plt.title('K-NN Distance Graph')
    plt.savefig(f'./features/knn_distance_{model_name}.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['lenet', 'alexnet', 'resnet18', 'googlenet', 'vgg11', 'efficientnet',
                                 'mnasnet', 'mobilenetv3', 'shufflenetv2', 'squeezenet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for t-SNE')
    parser.add_argument('--perplexity', type=float, default=30.0,
                        help='Perplexity parameter for t-SNE')
    args = parser.parse_args()

    if not os.path.exists('./features'):
        os.makedirs('./features')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=False, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # 加载模型
    if args.model == 'lenet':
        model = LeNet()
    elif args.model == 'alexnet':
        model = AlexNet()
    elif args.model == 'resnet18':
        model = ResNet18()
    elif args.model == 'googlenet':
        model = GoogLeNet()
    elif args.model == 'vgg11':
        model = VGG11()
    elif args.model == 'mnasnet':
        model = MNASNet()
    elif args.model == 'mobilenetv3':
        model = MobileNetV3Small()
    elif args.model == 'shufflenetv2':
        model = ShuffleNetV2()
    elif args.model == 'squeezenet':
        model = SqueezeNet()
    elif args.model == 'efficientnet':
        model = EfficientNet()
    else:
        raise ValueError('Invalid model name')

    # 读取results.csv, 找到Model,Scheduler,Test Accuray三列; 选择当前model,看哪个Test Accuracy更高,选择相应的scheduler
    results = pd.read_csv('./results.csv')
    results = results[results['Model'] == args.model]
    results = results[['Model', 'Scheduler', 'Test Accuracy']]
    results = results.sort_values(by='Test Accuracy', ascending=False)
    lr_scheduler = results.iloc[0]['Scheduler']

    model.load_state_dict(torch.load(f'./states/best_model_{args.model}_{lr_scheduler}.pth', weights_only=False))
    feature_extractor = FeatureExtractor(model).to(device)

    # 提取特征
    print("Extracting features...")
    features, labels = extract_features(feature_extractor, data_loader, device)

    # t-SNE降维
    print("Performing t-SNE...")
    tsne = TSNE(n_components=args.n_components, perplexity=args.perplexity)
    features_embedded = tsne.fit_transform(features)

    # K-means聚类
    print("Performing K-means clustering...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans_clusters = kmeans.fit_predict(features)

    # 使用ARI评估的DBSCAN聚类
    print("Finding best DBSCAN parameters...")
    plot_knn_distance(features, args.model)
    best_params, dbscan_clusters = find_best_dbscan_params(features_embedded, labels)
    eps, min_samples = best_params
    print(f"Best DBSCAN parameters: eps={eps:.3f}, min_samples={min_samples}")

    # 计算并输出最终聚类结果的评估指标
    kmeans_ari = adjusted_rand_score(labels, kmeans_clusters)
    dbscan_ari = adjusted_rand_score(labels, dbscan_clusters)
    print(f"K-means ARI: {kmeans_ari:.3f}")
    print(f"DBSCAN ARI: {dbscan_ari:.3f}")

    # 统计DBSCAN聚类结果
    n_clusters = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    n_noise = list(dbscan_clusters).count(-1)
    print(f'DBSCAN found {n_clusters} clusters and {n_noise} noise points')

    # 可视化
    plot_features(features_embedded, labels, kmeans_clusters, dbscan_clusters, args.model)
    print("Results saved.")

    # save ari to csv. if not exists, create a new file and write the header
    if not os.path.exists('./features/ari.csv'):
        with open('./features/ari.csv', 'w') as f:
            f.write('model,kmeans_ari,dbscan_ari\n')

    with open('./features/ari.csv', 'a') as f:
        f.write(f'{args.model},{kmeans_ari:.4f},{dbscan_ari:.4f}\n')


if __name__ == '__main__':
    main()
