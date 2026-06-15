import { useState, useRef, useEffect } from 'react';
import './PipelineRunner.css';

const MOCK_LOGS = [
  { text: "> source .venv/bin/activate", delay: 500, type: "info" },
  { text: "> python src/models/train.py --experiments track_level", delay: 800, type: "info" },
  { text: "[INFO] Scanning for radar sequences in data/data/ ...", delay: 600, type: "info" },
  { text: "[INFO] Found 50 sequence(s).", delay: 300, type: "success" },
  { text: "[INFO] Loading aggregated track features (radar_track_features.csv)...", delay: 1200, type: "info" },
  { text: "[INFO] 2,225 tracks loaded.", delay: 400, type: "success" },
  { text: "[INFO] Applying SimpleImputer and StandardScaler...", delay: 600, type: "info" },
  { text: "[INFO] Training Random Forest model (n_estimators=150)...", delay: 400, type: "info" },
  { text: "[==================================================] 100%", delay: 600, type: "success" },
  { text: "[INFO] Random Forest converged.", delay: 300, type: "success" },
  { text: "[INFO] Training XGBoost model...", delay: 400, type: "info" },
  { text: "[==================>                               ] 35%", delay: 800, type: "info" },
  { text: "[====================================>             ] 72%", delay: 800, type: "info" },
  { text: "[==================================================] 100%", delay: 1500, type: "success" },
  { text: "[INFO] XGBoost converged.", delay: 300, type: "success" },
  { text: "[INFO] Evaluating Macro F1 Score...", delay: 800, type: "info" },
  { text: "[SUCCESS] Pipeline execution completed in 42.1 seconds.", delay: 500, type: "success" }
];

export default function PipelineRunner() {
  const [logs, setLogs] = useState([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const terminalRef = useRef(null);

  const executePipeline = async () => {
    setIsExecuting(true);
    setShowResults(false);
    setLogs([]);

    let currentLogs = [];
    for (const log of MOCK_LOGS) {
      await new Promise(resolve => setTimeout(resolve, log.delay));
      currentLogs = [...currentLogs, log];
      setLogs(currentLogs);
    }

    setTimeout(() => {
      setIsExecuting(false);
      setShowResults(true);
    }, 1000);
  };

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="pipeline-runner">
      <h2>Execute Machine Learning Pipeline</h2>
      <p>Simulate running the data ingestion, feature engineering, and model training scripts locally.</p>
      
      <div className="pipeline-controls">
        <button 
          className="execute-btn" 
          onClick={executePipeline}
          disabled={isExecuting}
        >
          {isExecuting ? 'Executing...' : '▶ Run Pipeline'}
        </button>
      </div>

      <div className="terminal-window">
        <div className="terminal-header">
          <div className="terminal-btn close"></div>
          <div className="terminal-btn minimize"></div>
          <div className="terminal-btn maximize"></div>
          <div className="terminal-title">bash - python src/models/train.py</div>
        </div>
        <div className="terminal-body" ref={terminalRef}>
          {logs.map((log, index) => (
            <p key={index} className={`log-line log-${log.type}`}>
              {log.text}
            </p>
          ))}
          {isExecuting && <div className="cursor"></div>}
        </div>
      </div>

      {showResults && (
        <div className="results-panel">
          <h3>Execution Results</h3>
          <p>The models have been successfully trained and evaluated against the test set.</p>
          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Top Model</h4>
              <div className="metric-value">XGBoost</div>
            </div>
            <div className="metric-card">
              <h4>Macro F1 Score</h4>
              <div className="metric-value">53.7%</div>
            </div>
            <div className="metric-card">
              <h4>Total Processed</h4>
              <div className="metric-value">Track-Level Aggregation</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
