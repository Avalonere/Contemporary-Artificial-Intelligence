// App.js
import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import IcePuzzle from './components/IcePuzzle';
import PathFinderVisualization from './components/PyramidPuzzle';

const App = () => {
  return (
    <Router>
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6' }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/ice-puzzle" element={<IcePuzzle />} />
          <Route path="/pyramid-puzzle" element={<PathFinderVisualization />} />
        </Routes>
      </div>
    </Router>
  );
};

const HomePage = () => {
  const containerStyle = {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '64px 16px'
  };

  const titleStyle = {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: '48px'
  };

  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '32px',
    maxWidth: '800px',
    margin: '0 auto'
  };

  const cardStyle = {
    padding: '24px',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
    textDecoration: 'none',
    color: 'inherit',
    transition: 'transform 0.2s, box-shadow 0.2s'
  };

  const cardTitleStyle = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
    marginBottom: '16px'
  };


  return (
    <div style={containerStyle}>
      <h1 style={titleStyle}>AI Project2: A* Algorithm</h1>
      <div style={gridStyle}>
        <Link to="/ice-puzzle" style={cardStyle}>
          <h2 style={cardTitleStyle}>Q1: 冰雪魔方的冰霜之道</h2>

        </Link>
        <Link to="/pyramid-puzzle" style={cardStyle}>
          <h2 style={cardTitleStyle}>Q2: 杰克的金字塔探险</h2>

        </Link>
      </div>
    </div>
  );
};


export default App;
