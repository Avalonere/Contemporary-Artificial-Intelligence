import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { MinPriorityQueue } from '@datastructures-js/priority-queue';

const TEST_CASES = [
  '150732684',
  '135720684',
  '105732684',
  '015732684',
  '135782604',
  '715032684',
];

const GOAL_STATE = '135702684';

class PuzzleState {
  constructor(state, gScore = 0, previous = null, swapIndex = null) {
    this.state = state;          // 当前状态字符串
    this.gScore = gScore;        // 从初始状态到当前状态的实际代价
    this.previous = previous;    // 前一个状态
    //this.lastMove = lastMove;    // 上一步移动方向
    this.swapIndex = swapIndex; // 与空白块交换的位置索引
    this.hScore = this.calculateHeuristic(); // 启发式评估值
    this.fScore = this.gScore + this.hScore; // 总评估值 f = g + h
  }

  // 计算曼哈顿距离作为启发式函数
  calculateHeuristic() {
    let distance = 0;
    for (let i = 0; i < 9; i++) {
      if (this.state[i] !== '0') {
        const currentPos = i;
        const targetPos = GOAL_STATE.indexOf(this.state[i]);
        // 计算当前位置和目标位置的行列坐标
        const currentRow = Math.floor(currentPos / 3);
        const currentCol = currentPos % 3;
        const targetRow = Math.floor(targetPos / 3);
        const targetCol = targetPos % 3;
        // 计算曼哈顿距离
        distance += Math.abs(currentRow - targetRow) + Math.abs(currentCol - targetCol);
      }
    }
    return distance;
  }

  // 获取所有可能的相邻状态
  getNeighbors() {
    const neighbors = [];
    const zeroPos = this.state.indexOf('0');
    const zeroRow = Math.floor(zeroPos / 3);
    const zeroCol = zeroPos % 3;
    const moves = [[-1, 0], [1, 0], [0, -1], [0, 1]];
    const directions = ['上', '下', '左', '右'];

    moves.forEach(([dx, dy], index) => {
      const newRow = zeroRow + dx;
      const newCol = zeroCol + dy;
      if (newRow >= 0 && newRow < 3 && newCol >= 0 && newCol < 3) {
        const newPos = newRow * 3 + newCol;
        const swappedNumber = this.state[newPos]; // 记录被交换的数字
        const newState = this.state.split('');
        [newState[zeroPos], newState[newPos]] = [newState[newPos], newState[zeroPos]];
        neighbors.push({
          state: new PuzzleState(
            newState.join(''),
            this.gScore + 1,
            this,
            //directions[index],
            swappedNumber  // 存储被交换的数字而不是位置
          ),
          direction: directions[index],
        });
      }
    });
    return neighbors;
  }
}

// A*搜索算法实现
const solvePuzzle = (initialState) => {
  const start = new PuzzleState(initialState);
  const openSet = new MinPriorityQueue((state) => state.fScore);
  openSet.enqueue(start);
  const closedSet = new Set();
  const allStates = new Map();
  const pathChoices = new Map();
  const openSetStates = new Map();

 while (!openSet.isEmpty()) {
    const current = openSet.dequeue();

    if (current.state === GOAL_STATE) {
      const path = [];
      let temp = current;
      while (temp) {
        const choices = pathChoices.get(temp.state) || [];
        const openSetForState = openSetStates.get(temp.state) || [];
        path.unshift({
          state: temp.state,
          costs: allStates.get(temp.state),
          choices: choices,
          //lastMove: temp.lastMove,
          swappedNumber: temp.swapIndex, // 存储交换的数字
          openSet: openSetForState,
        });
        temp = temp.previous;
      }
      return path;
    }

    closedSet.add(current.state);

    // 探索相邻状态
    const neighbors = current.getNeighbors();
    const currentChoices = [];

    for (const { state: neighbor, direction } of neighbors) {
      if (closedSet.has(neighbor.state)) continue;

      currentChoices.push({
        state: neighbor.state,
        direction,
        costs: {
          gScore: neighbor.gScore,
          hScore: neighbor.hScore,
          fScore: neighbor.fScore,
        },
        swapIndex: neighbor.swapIndex,
      });

        const existingNode = openSet.toArray().find((node) => node.state === neighbor.state);
        if (existingNode) {
          if (neighbor.fScore < existingNode.fScore) {
            openSet.remove(existingNode);
            openSet.enqueue(neighbor);
          }
        } else {
          openSet.enqueue(neighbor);
        }
    }

    if (currentChoices.length > 0) {
      pathChoices.set(current.state, currentChoices);
    }

    // 记录当前openSet状态
    const currentOpenSetStates = Array.from(openSet.toArray()).map((state) => ({
      state: state.state,
      costs: {
        gScore: state.gScore,
        hScore: state.hScore,
        fScore: state.fScore,
      },
      swappedNumber: state.swapIndex  // 存储交换的数字
    }));
    openSetStates.set(current.state, currentOpenSetStates);
  }
  return null;
};


const PuzzleGrid = ({ state, highlightIndex }) => {
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px',
    padding: '16px',
    backgroundColor: '#f3f4f6',
    borderRadius: '8px',
    width: 'fit-content',
    margin: '0 auto',
  };

  const cellStyle = (value, index) => ({
    width: '80px',
    height: '80px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor:
      value === '0' ? '#e5e7eb' : index === highlightIndex ? '#f59e0b' : '#3b82f6',
    color: value === '0' ? 'transparent' : 'white',
    fontSize: '32px',
    fontWeight: 'bold',
    borderRadius: '8px',
    transition: 'all 0.3s',
  });

  return (
    <div>
      <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '16px', textAlign: 'center' }}>
        当前状态
      </h3>
      <div style={gridStyle}>
        {state.split('').map((value, index) => (
          <div key={index} style={cellStyle(value, index)}>
            {value}
          </div>
        ))}
      </div>
    </div>
  );
};

const SmallPuzzleGrid = ({ state, swappedNumber }) => {
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '4px',
    padding: '8px',
    backgroundColor: '#f3f4f6',
    borderRadius: '8px',
    width: 'fit-content',
    margin: '0 auto',
  };

  const cellStyle = (value) => ({
    width: '40px',
    height: '40px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor:
      value === '0' ? '#e5e7eb' :
      value === swappedNumber ? '#ec4899' : // 高亮显示交换的数字
      '#3b82f6',
    color: value === '0' ? 'transparent' : 'white',
    fontSize: '16px',
    fontWeight: 'bold',
    borderRadius: '4px',
  });

  return (
    <div style={gridStyle}>
      {state.split('').map((value, index) => (
        <div key={index} style={cellStyle(value)}>
          {value}
        </div>
      ))}
    </div>
  );
};

const MoveHistory = ({ solution, currentStep }) => {
  const containerStyle = {
    backgroundColor: 'white',
    borderRadius: '8px',
    padding: '16px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    maxHeight: '600px',
    overflowX: 'auto',
  };

  const historyStyle = {
    display: 'flex',
    flexDirection: 'row',
    gap: '16px',
    alignItems: 'center',
  };

  const stepStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  };

  const arrowStyle = {
    fontSize: '24px',
    color: '#666',
  };

  // 改为将箭头放在两个矩阵之间
 const historyElements = [];
  for (let i = 0; i <= currentStep; i++) {
    historyElements.push(
      <div key={`grid-${i}`} style={stepStyle}>
        <SmallPuzzleGrid
          state={solution[i].state}
          swappedNumber={solution[i].swappedNumber}
        />
      </div>
    );
    if (i < currentStep) {
      historyElements.push(
        <span key={`arrow-${i}`} style={arrowStyle}>
          →
        </span>
      );
    }
  }

  return (
    <div style={containerStyle}>
      <h3 style={{
        fontSize: '1.25rem',
        fontWeight: 'bold',
        marginBottom: '16px',
        textAlign: 'center',
      }}>
        状态变化历史
      </h3>
      <div style={historyStyle}>{historyElements}</div>
    </div>
  );
};
const OpenSetDisplay = ({ openSet }) => {
  if (!openSet || openSet.length === 0) return null;

  const containerStyle = {
    textAlign: 'center',
    marginTop: '32px',
  };

  const choicesStyle = {
    display: 'flex',
    justifyContent: 'center',
    gap: '16px',
    flexWrap: 'wrap',
  };

  const gridContainerStyle = (isMin) => ({
    border: isMin ? '3px solid red' : '1px solid #d1d5db',
    borderRadius: '8px',
    padding: '8px',
    position: 'relative',
  });

  // 定义颜色
  const colors = {
    f: '#f59e0b', // 橙色
    g: '#8b5cf6', // 紫色
    h: '#10b981', // 绿色
  };

  const costStyle = {
    marginTop: '8px',
    fontSize: '14px',
    textAlign: 'center',
  };
  const minFScore = Math.min(...openSet.map((state) => state.costs.fScore));

  return (
    <div style={containerStyle}>
      <h3 style={{ fontSize: '1.25rem', fontWeight: 'bold', marginBottom: '16px' }}>
        OpenSet 状态
      </h3>
      <div style={choicesStyle}>
        {openSet.map((state, index) => (
          <div key={index} style={gridContainerStyle(state.costs.fScore === minFScore)}>
            <SmallPuzzleGrid
              state={state.state}
              swappedNumber={state.swappedNumber}
            />
            <div style={costStyle}>
              <span style={{ color: colors.f, fontWeight: 'bold' }}>{state.costs.fScore}</span>
              {' = '}
              <span style={{ color: colors.g }}>{state.costs.gScore}</span>
              {' + '}
              <span style={{ color: colors.h }}>{state.costs.hScore}</span>
            </div>
          </div>
        ))}
      </div>
      {/* ... 其他代码保持不变 ... */}
    </div>
  );
};

const IcePuzzle = () => {
  const [selectedCase, setSelectedCase] = useState(0);
  const [solution, setSolution] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const newSolution = solvePuzzle(TEST_CASES[selectedCase]);
    setSolution(newSolution);
    setCurrentStep(0);
  }, [selectedCase]);

  if (!solution) return <div>Loading...</div>;
  const currentOpenSet = solution[currentStep]?.openSet;
  const isOpenSetAvailable = currentOpenSet && currentOpenSet.length > 0;

  // 根据是否有可选移动动态调整布局
  const gridContainerStyle = {
    display: 'grid',
    gridTemplateColumns: isOpenSetAvailable ? '1fr 1fr 1fr' : '1fr 1fr',
    gap: '32px',
    alignItems: 'flex-start',
    justifyContent: 'center',
  };

  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '32px 16px',
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '32px',
  };

  const titleStyle = {
    fontSize: '2rem',
    fontWeight: 'bold',
  };

  const buttonStyle = {
    padding: '8px 16px',
    backgroundColor: '#6b7280',
    color: 'white',
    borderRadius: '8px',
    textDecoration: 'none',
    transition: 'background-color 0.2s',
  };

  const selectContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    marginBottom: '32px',
  };

  const selectStyle = {
    padding: '8px 16px',
    borderRadius: '8px',
    border: '1px solid #d1d5db',
  };

  const controlsStyle = {
    display: 'flex',
    justifyContent: 'center',
    gap: '16px',
    marginTop: '32px',
  };

  const controlButtonStyle = (disabled) => ({
    padding: '12px 24px',
    backgroundColor: disabled ? '#d1d5db' : '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: disabled ? 'default' : 'pointer',
  });

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <h1 style={titleStyle}>Q1: 冰雪魔方的冰霜之道</h1>
        <Link to="/" style={buttonStyle}>
          返回首页
        </Link>
      </div>

      <div style={selectContainerStyle}>
        <label style={{ fontSize: '1.125rem' }}>
          选择测试用例：
          <select
            value={selectedCase}
            onChange={(e) => setSelectedCase(Number(e.target.value))}
            style={selectStyle}
          >
            {TEST_CASES.map((testCase, index) => (
              <option key={index} value={index}>
                Case {index + 1}: {testCase}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={gridContainerStyle}>
        {/* 左侧：当前状态和控制按钮 */}
        <div>
          <PuzzleGrid
            state={solution[currentStep].state}
            highlightIndex={solution[currentStep].swapIndex}
          />

          <div style={controlsStyle}>
            <button
              onClick={() => setCurrentStep(0)}
              disabled={currentStep === 0}
              style={controlButtonStyle(currentStep === 0)}
            >
              重置
            </button>
            <button
              onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
              disabled={currentStep === 0}
              style={controlButtonStyle(currentStep === 0)}
            >
              上一步
            </button>
            <button
              onClick={() => setCurrentStep(Math.min(solution.length - 1, currentStep + 1))}
              disabled={currentStep === solution.length - 1}
              style={controlButtonStyle(currentStep === solution.length - 1)}
            >
              下一步
            </button>
          </div>

          <div style={{ textAlign: 'center', color: '#666', marginTop: '16px' }}>
            步骤 {currentStep} / {solution.length - 1}
          </div>
        </div>

        {/* 中间：OpenSet 状态（根据是否有 openset 决定是否显示） */}
        {isOpenSetAvailable && (
          <div>
            <OpenSetDisplay openSet={currentOpenSet} />
          </div>
        )}

        {/* 右侧：移动历史 */}
        <div>
          <MoveHistory solution={solution} currentStep={currentStep} />
        </div>
      </div>
    </div>
  );
};

export default IcePuzzle;
