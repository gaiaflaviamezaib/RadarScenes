import { useState } from 'react';
import './ModelComparison.css';

const metrics = [
  { model: "XGBoost",              accuracy: 0.776, macroF1: 0.537 },
  { model: "Random Forest",        accuracy: 0.758, macroF1: 0.521 },
  { model: "Logistic Regression",  accuracy: 0.711, macroF1: 0.417 },
  { model: "Gaussian Naive Bayes", accuracy: 0.535, macroF1: 0.299 },
];


export default function ModelComparison() {
  const [sortConfig, setSortConfig] = useState({ key: "macroF1", direction: "descending" });

  const sortedMetrics = [...metrics].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? -1 : 1;
    }
    if (a[sortConfig.key] > b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? 1 : -1;
    }
    return 0;
  });

  const requestSort = (key) => {
    let direction = 'descending';
    if (sortConfig.key === key && sortConfig.direction === 'descending') {
      direction = 'ascending';
    }
    setSortConfig({ key, direction });
  };

  return (
    <section className="section" id="models">
      <div className="section-head">
        <p className="eyebrow">Supervised Learning</p>
        <h2>Model Comparison Leaderboard</h2>
        <p>
          Final benchmark on 50 sequences using track-level aggregated features. 
          Click the column headers to sort by Accuracy or Macro F1 Score.
        </p>
      </div>

      <div className="model-controls">
        <span className="experiment-badge">Track-Level Aggregation — 50 Sequences</span>
      </div>

      <div className="table-responsive">
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th>Model</th>
              <th 
                className={sortConfig.key === 'accuracy' ? 'sort-active' : ''}
                onClick={() => requestSort('accuracy')}
              >
                Accuracy {sortConfig.key === 'accuracy' ? (sortConfig.direction === 'descending' ? '↓' : '↑') : ''}
              </th>
              <th 
                className={sortConfig.key === 'macroF1' ? 'sort-active' : ''}
                onClick={() => requestSort('macroF1')}
              >
                Macro F1 Score {sortConfig.key === 'macroF1' ? (sortConfig.direction === 'descending' ? '↓' : '↑') : ''}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedMetrics.map((row, index) => (
              <tr key={index} className={index === 0 && sortConfig.key === 'macroF1' && sortConfig.direction === 'descending' ? 'top-row' : ''}>
                <td className="model-name">{row.model}</td>
                <td>{(row.accuracy * 100).toFixed(1)}%</td>
                <td className="highlight-metric">{(row.macroF1 * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
