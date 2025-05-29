import React, { useState, useEffect } from 'react';
import { MinPriorityQueue } from '@datastructures-js/priority-queue';

const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '20px',
  },
  controlPanel: {
    marginBottom: '20px',
  },
  select: {
    padding: '8px 12px',
    marginRight: '10px',
    borderRadius: '4px',
    border: '1px solid #ccc',
  },
  button: {
    padding: '8px 16px',
    marginRight: '10px',
    backgroundColor: '#2563eb',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  buttonDisabled: {
    backgroundColor: '#93c5fd',
    cursor: 'not-allowed',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
  },
  card: {
    border: '1px solid #e5e7eb',
    borderRadius: '8px',
    backgroundColor: 'white',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  },
  cardHeader: {
    padding: '16px 20px',
    borderBottom: '1px solid #e5e7eb',
  },
  cardTitle: {
    margin: 0,
    fontSize: '18px',
    fontWeight: 'bold',
  },
  cardContent: {
    padding: '20px',
  },
  queueItem: {
    padding: '12px',
    backgroundColor: '#f8fafc',
    borderRadius: '4px',
    marginBottom: '8px',
    border: '1px solid #e2e8f0',
  },
  queueItemText: {
    margin: '4px 0',
    fontSize: '14px',
  },
  pathList: {
    marginTop: '20px',
  },
  pathItem: {
    margin: '8px 0',
  },
  sectionTitle: {
    fontSize: '16px',
    fontWeight: 'bold',
    marginBottom: '12px',
  }
};

// Test cases data
const TEST_CASES = [
  {
    n: 5, m: 6, k: 3,
    edges: [[1,2,1], [1,3,4], [2,4,3], [3,4,2], [3,5,1], [4,5,2]]
  },
  {
    n: 5, m: 6, k: 4,
    edges: [[1,2,1], [1,3,1], [2,4,2], [2,5,2], [3,4,2], [3,5,2]]
  },
  // Other test cases can be added here
];

class PathFinder {
  constructor(n, edges) {
    this.n = n;
    this.graph = new Map();

    for (let i = 1; i <= n; i++) {
      this.graph.set(i, []);
    }

    edges.forEach(([x, y, d]) => {
      this.graph.get(x).push({ node: y, weight: d });
    });
  }

  getMinEdge(node) {
    if (node === this.n) return 0;
    if (!this.graph.get(node).length) return Infinity;

    const minDist = Math.min(...this.graph.get(node).map(({weight}) => weight + 1));

    for (const {node: nextNode, weight} of this.graph.get(node)) {
      if (nextNode === this.n) return Math.min(minDist, weight);
    }

    return minDist;
  }

  *aStarGenerator(k) {
    const paths = [];
    const pq = new MinPriorityQueue(item => item.estimatedTotal);
    const seen = new Set();

    // 初始化起点的邻接状态
    const startNode = 1;
    const initialOptions = [];
    for (const {node: nextNode, weight} of this.graph.get(startNode)) {
      if (nextNode > startNode) {
        const newPath = [startNode, nextNode];
        const newDist = weight;
        const estimatedTotal = newDist + this.getMinEdge(nextNode);
        initialOptions.push({
          estimatedTotal,
          currentDist: newDist,
          node: nextNode,
          path: newPath
        });
      }
    }

    // 将所有初始选项加入队列
    initialOptions.forEach(option => pq.enqueue(option));

    // 先yield初始状态
    yield {
      currentNode: startNode,
      currentPath: [startNode],
      queueState: initialOptions,
      pathsSoFar: []
    };

    while (!pq.isEmpty() && paths.length < k) {
      const current = pq.dequeue();
      const { currentDist, node, path } = current;

      // 生成当前节点的下一步选项
      const nextOptions = [];
      for (const {node: nextNode, weight} of this.graph.get(node)) {
        if (nextNode > node) {
          const newPath = [...path, nextNode];
          if (!seen.has(newPath.join(','))) {
            const newDist = currentDist + weight;
            const estimatedTotal = newDist + this.getMinEdge(nextNode);
            nextOptions.push({
              estimatedTotal,
              currentDist: newDist,
              node: nextNode,
              path: newPath
            });
          }
        }
      }

      // 将新选项加入队列
      nextOptions.forEach(option => pq.enqueue(option));

      if (node === this.n) {
        paths.push({ distance: currentDist, path });
      } else {
        const pathKey = path.join(',');
        if (!seen.has(pathKey)) {
          seen.add(pathKey);
        }
      }
      
      // 创建一个queuestate，包含nextOption和pq并去掉两个都有的
        const queueState = Array.from(pq.toArray()).map(item => ({ path: item.path, currentDist: item.currentDist, estimatedTotal: item.estimatedTotal }));

      // yield当前状态和下一步可能的选项
      yield {
        currentNode: node,
        currentPath: path,
        queueState: queueState,
        pathsSoFar: [...paths]
      };
    }

    // 填充剩余路径
    while (paths.length < k) {
      paths.push({ distance: -1, path: [] });
    }

    // 最终状态
    yield {
      currentNode: this.n,
      currentPath: paths[paths.length - 1].path,
      queueState: [],
      pathsSoFar: paths
    };
  }
}

const GraphVisualization = ({ n, edges, currentPath, currentNode }) => {
  const radius = 120;
  const nodePositions = new Map();
  const center = { x: 150, y: 150 };

  for (let i = 1; i <= n; i++) {
    const angle = ((i - 1) / n) * 2 * Math.PI - Math.PI / 2;
    nodePositions.set(i, {
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle)
    });
  }

  // 计算边权重标签的偏移位置，避免遮挡
  const getLabelPosition = (fromPos, toPos) => {
    const midX = (fromPos.x + toPos.x) / 2;
    const midY = (fromPos.y + toPos.y) / 2;

    // 计算垂直于边的方向向量
    const dx = toPos.x - fromPos.x;
    const dy = toPos.y - fromPos.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    const normalX = -dy / len;
    const normalY = dx / len;

    // 将标签沿垂直方向偏移
    const offset = 0;
    return {
      x: midX + normalX * offset,
      y: midY + normalY * offset
    };
  };

  return (
    <svg viewBox="0 0 300 300" style={{ width: '100%', height: '100%' }}>
      {/* 先绘制边 */}
      {edges.map(([from, to, weight], idx) => {
        const fromPos = nodePositions.get(from);
        const toPos = nodePositions.get(to);
        const isInPath = currentPath?.some((node, i) =>
          node === from && currentPath[i + 1] === to
        );

        const labelPos = getLabelPosition(fromPos, toPos);

        return (
          <g key={`edge-${idx}`}>
            <line
              x1={fromPos.x}
              y1={fromPos.y}
              x2={toPos.x}
              y2={toPos.y}
              stroke={isInPath ? "#2563eb" : "#94a3b8"}
              strokeWidth="2"
            />
            <circle
              cx={labelPos.x}
              cy={labelPos.y}
              r="10"
              fill="white"
              stroke="#94a3b8"
            />
            <text
              x={labelPos.x}
              y={labelPos.y}
              fill="#64748b"
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="12"
            >
              {weight}
            </text>
          </g>
        );
      })}

      {/* 后绘制节点，确保在最上层 */}
      {Array.from({ length: n }, (_, i) => i + 1).map(node => {
        const pos = nodePositions.get(node);
        const isActive = node === currentNode;
        const isInPath = currentPath?.includes(node);

        return (
          <g key={`node-${node}`}>
            <circle
              cx={pos.x}
              cy={pos.y}
              r="20"
              fill={isActive ? "#2563eb" : isInPath ? "#93c5fd" : "#fff"}
              stroke="#1e40af"
              strokeWidth="2"
            />
            <text
              x={pos.x}
              y={pos.y}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={isActive || isInPath ? "#fff" : "#000"}
              fontSize="14"
              fontWeight="bold"
            >
              {node}
            </text>
          </g>
        );
      })}
    </svg>
  );
};


const QueueStateView = ({ queueState }) => (
  <div>
    <h3 style={styles.sectionTitle}>Priority Queue State</h3>
    {queueState.map((item, idx) => (
        <div key={idx} style={styles.queueItem}>
          <div style={styles.queueItemText}>
            Path: <span style={{color: 'red'}}>{item.path.join(' → ')}</span>
          </div>
          <div style={styles.queueItemText}>
            Current Distance: <span style={{color: 'orange'}}>{item.currentDist}</span>
          </div>
          <div style={styles.queueItemText}>
            Heuristic: <span style={{color: 'purple'}}>{item.estimatedTotal - item.currentDist}</span>
          </div>
          <div style={styles.queueItemText}>
            Estimated Total: <span style={{color: 'green'}}>{item.estimatedTotal}</span>
          </div>
        </div>
    ))}
  </div>
);

const PathFinderVisualization = () => {
  const [selectedCase, setSelectedCase] = useState(0);
  const [currentState, setCurrentState] = useState(null);
  const [generator, setGenerator] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const initializePathFinder = () => {
    const testCase = TEST_CASES[selectedCase];
    const finder = new PathFinder(testCase.n, testCase.edges);
    setGenerator(finder.aStarGenerator(testCase.k));
    setCurrentState(null);
    setIsRunning(false);
  };

  const step = () => {
    if (!generator) return;

    const result = generator.next();
    if (!result.done) {
      setCurrentState(result.value);
    } else {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    initializePathFinder();
  }, [selectedCase]);

  return (
    <div style={styles.container}>
      <div style={styles.controlPanel}>
        <select
          value={selectedCase}
          onChange={(e) => setSelectedCase(parseInt(e.target.value))}
          style={styles.select}
        >
          {TEST_CASES.map((_, idx) => (
            <option key={idx} value={idx}>
              Test Case {idx + 1}
            </option>
          ))}
        </select>
        <button
          onClick={step}
          disabled={!generator || isRunning}
          style={{...styles.button, ...((!generator || isRunning) && styles.buttonDisabled)}}
        >
          Step
        </button>
        <button onClick={initializePathFinder} style={styles.button}>
          Reset
        </button>
      </div>

      <div style={styles.grid}>
        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <h2 style={styles.cardTitle}>Graph Visualization</h2>
          </div>
          <div style={styles.cardContent}>
            <GraphVisualization
              n={TEST_CASES[selectedCase].n}
              edges={TEST_CASES[selectedCase].edges}
              currentPath={currentState?.currentPath}
              currentNode={currentState?.currentNode}
            />
          </div>
        </div>

        <div style={styles.card}>
          <div style={styles.cardHeader}>
            <h2 style={styles.cardTitle}>Algorithm State</h2>
          </div>
          <div style={styles.cardContent}>
            <QueueStateView queueState={currentState?.queueState || []} />
            {currentState?.pathsSoFar.length > 0 && (
              <div style={styles.pathList}>
                <h3 style={styles.sectionTitle}>Found Paths</h3>
                {currentState.pathsSoFar.map((path, idx) => (
                    <div key={idx} style={styles.pathItem}>
                      Path {idx + 1}: <span style={{color: 'red'}}>{path.path.join(' → ')}</span> (Distance: <span
                        style={{color: 'orange'}}>{path.distance}</span>)
                    </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PathFinderVisualization;