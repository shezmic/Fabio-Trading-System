import React, { useEffect, useState } from 'react';
import Chart from './components/Chart';
import WebSocketClient from './services/WebSocketClient';

function App() {
  const [marketData, setMarketData] = useState([]);
  const [signals, setSignals] = useState([]);
  
  useEffect(() => {
    const ws = new WebSocketClient('ws://localhost:8000/ws'); // Placeholder URL
    
    ws.on('trade.tick', (data) => {
      // Update chart data
      // setMarketData(prev => [...prev, data]);
    });
    
    ws.on('strategy.signal', (data) => {
      setSignals(prev => [data, ...prev]);
    });
    
    ws.connect();
    
    return () => ws.disconnect();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Fabio Trading System</h1>
      </header>
      <main>
        <div className="chart-container">
          <Chart data={marketData} />
        </div>
        <div className="signals-panel">
          <h2>Recent Signals</h2>
          <ul>
            {signals.map((s, i) => (
              <li key={i}>{s.symbol} - {s.direction} ({s.grade})</li>
            ))}
          </ul>
        </div>
      </main>
    </div>
  );
}

export default App;
