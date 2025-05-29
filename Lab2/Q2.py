import heapq
from collections import defaultdict
import random
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class PathFinder:
    def __init__(self, n, edges):
        self.n = n
        self.graph = defaultdict(list)
        for x, y, d in edges:
            self.graph[x].append((y, d))

    def get_min_edge(self, node):
        if node == self.n:
            return 0

        if not self.graph[node]:
            return float('inf')

        min_dist = min([weight + 1 for _, weight in self.graph[node]])

        for next_node, weight in self.graph[node]:
            if next_node == self.n:
                return min(min_dist, weight)

        return min_dist


    def dijkstra_from_bottom(self):
        # 从底向上的Dijkstra
        distances = {i: float('inf') for i in range(1, self.n + 1)}
        distances[self.n] = 0
        pq = [(0, self.n)]

        reverse_graph = defaultdict(list)
        for u in self.graph:
            for v, d in self.graph[u]:
                reverse_graph[v].append((u, d))

        while pq:
            dist, node = heapq.heappop(pq)
            if dist > distances[node]:
                continue

            for next_node, weight in reverse_graph[node]:
                if distances[next_node] > dist + weight:
                    distances[next_node] = dist + weight
                    heapq.heappush(pq, (int(distances[next_node]), next_node))

        return distances

    def a_star(self, k, heuristic_type=1):
        start_time = time.time()
        dijkstra_time = 0

        if heuristic_type == 2:
            dijkstra_start = time.time()
            bottom_up_distances = self.dijkstra_from_bottom()
            dijkstra_time = time.time() - dijkstra_start

        def heuristic(node):
            if heuristic_type == 1:
                return self.get_min_edge(node)
            else:
                return bottom_up_distances[node]

        paths = []
        pq = [(heuristic(1), 0, 1, [1])]  # (estimated_total, current_distance, node, path)
        seen = set()

        while pq and len(paths) < k:
            _, current_dist, node, path = heapq.heappop(pq)

            if node == self.n:
                paths.append((current_dist, path))
                continue

            path_tuple = tuple(path)
            seen.add(path_tuple)

            for next_node, weight in self.graph[node]:
                if next_node > node:  # 只能向下移动
                    new_path = path + [next_node]
                    if tuple(new_path) in seen:
                        continue
                    new_dist = current_dist + weight
                    estimated_total = new_dist + heuristic(next_node)
                    heapq.heappush(pq, (estimated_total, new_dist, next_node, new_path))


        while len(paths) < k:
            paths.append((-1, []))

        return [dist for dist, _ in paths], time.time() - start_time, dijkstra_time, [path for _, path in paths]


def generate_test_case(n, m, min_weight=1):
    edges = []
    edge_set = set()

    max_weight = n // 2

    while len(edges) < m:
        x = random.randint(1, n - 1)
        y = random.randint(x + 1, n)
        if (x, y) not in edge_set:
            edge_set.add((x, y))
            edges.append([x, y, random.randint(min_weight, max_weight)])

    return edges


def run_performance_test():
    scale_factors = [2**i for i in range(15, 21)]  # 16384 to 2097152
    results = []

    for n in tqdm.tqdm(desc='Running performance tests', unit='nodes', iterable=scale_factors):
        m = n * 2  # 边的数量为节点数的两倍
        k = n // 20  # 5% of nodes

        scale_result = {
            "n": n,
            "h1_times": [],
            "h2_times": [],
            "dijkstra_times": []
        }

        for _ in tqdm.tqdm(range(100)):
            edges = generate_test_case(n, m)
            finder = PathFinder(n, edges)

            # 使用启发式1
            _, time_h1, _, _ = finder.a_star(k, 1)
            scale_result["h1_times"].append(time_h1)

            # 使用启发式2
            _, time_h2, dijkstra_time, _ = finder.a_star(k, 2)
            scale_result["h2_times"].append(time_h2)
            scale_result["dijkstra_times"].append(dijkstra_time)

        results.append(scale_result)

    # 保存结果
    with open('performance_results.json', 'w') as f:
        json.dump(results, f)

    return results


def plot_performance(results):
    # 提取规模
    scale_factors = [r["n"] for r in results]

    # 计算平均值和标准差
    h1_means = [np.mean(r["h1_times"]) for r in results]
    h2_means = [np.mean(r["h2_times"]) for r in results]
    dijkstra_means = [np.mean(r["dijkstra_times"]) for r in results]

    h1_stds = [np.std(r["h1_times"]) for r in results]
    h2_stds = [np.std(r["h2_times"]) for r in results]
    dijkstra_stds = [np.std(r["dijkstra_times"]) for r in results]

    # 设置style
    plt.style.use('seaborn-v0_8-pastel')
    plt.figure(figsize=(12, 8))
    plt.errorbar(scale_factors, h1_means, yerr=h1_stds, label='Heuristic 1', marker='o')
    plt.errorbar(scale_factors, h2_means, yerr=h2_stds, label='Heuristic 2', marker='s')
    plt.errorbar(scale_factors, dijkstra_means, yerr=dijkstra_stds, label='Dijkstra Time', marker='^')

    # x用log2尺度
    plt.xscale('log', base=2)
    plt.xlabel('Number of Nodes (N) in Log2 Scale')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison of Different Heuristics')
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    return plt


# 测试主函数
def main():
    # 测试用例
    test_cases = [
    # 测试用例 1
    [
        [5, 6, 3],
        [1, 2, 1], [1, 3, 4], [2, 4, 3],
        [3, 4, 2], [3, 5, 1], [4, 5, 2]
    ],
    # 测试用例 2
    [
        [5, 6, 4],
        [1, 2, 1], [1, 3, 1], [2, 4, 2],
        [2, 5, 2], [3, 4, 2], [3, 5, 2]
    ],
    # 测试用例 3
    [
        [6, 9, 4],
        [1, 2, 1], [1, 3, 3], [2, 4, 2],
        [2, 5, 3], [3, 6, 1], [4, 6, 3],
        [5, 6, 3], [1, 6, 8], [2, 6, 4]
    ],
    # 测试用例 4
    [
        [7, 12, 6],
        [1, 2, 1], [1, 3, 3], [2, 4, 2],
        [2, 5, 3], [3, 6, 1], [4, 7, 3],
        [5, 7, 1], [6, 7, 2], [1, 7, 10],
        [2, 6, 4], [3, 4, 2], [4, 5, 1]
    ],
    # 测试用例 5
    [
        [5, 8, 7],
        [1, 2, 1], [1, 3, 3], [2, 4, 1],
        [2, 5, 3], [3, 4, 2], [3, 5, 2],
        [1, 4, 3], [1, 5, 4]
    ],
    # 测试用例 6
    [
        [6, 10, 8],
        [1, 2, 1], [1, 3, 2], [2, 4, 2],
        [2, 5, 3], [3, 6, 3], [4, 6, 3],
        [5, 6, 1], [1, 6, 8], [2, 6, 5],
        [3, 4, 1]
    ]
    ]

    for case in test_cases:
        n, m, k = case[0]
        edges = case[1:]

        finder = PathFinder(n, edges)
        print(f"Test case: n={n}, m={m}, k={k}")
        # 使用启发式1
        paths1, time1, _, detailed_path1 = finder.a_star(k, 1)
        print(f"Heuristic 1 results: {paths1}")
        print(f"Path: {detailed_path1}")
        #print(f"Time taken: {time1:.4f}s")

        # 使用启发式2
        paths2, time2, dijkstra_time, detailed_path2 = finder.a_star(k, 2)
        print(f"Heuristic 2 results: {paths2}")
        print(f"Path: {detailed_path2}")
        #print(f"Time taken: {time2:.4f}s (Dijkstra time: {dijkstra_time:.4f}s)")

        print('\n')
        assert paths1 == paths2, "Different heuristics should give the same results!"

    # 运行性能测试
    results = run_performance_test()
    # read from json
    # with open('performance_results.json', 'r') as f:
    #     results = json.load(f)
    plt = plot_performance(results)
    plt.savefig('performance_comparison.png')
    plt.close()


if __name__ == "__main__":
    main()
