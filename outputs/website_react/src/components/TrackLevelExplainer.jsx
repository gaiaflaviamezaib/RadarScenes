import { useState } from 'react';
import './TrackLevelExplainer.css';

export default function TrackLevelExplainer() {
  const [view, setView] = useState('point');

  return (
    <section className="section" id="track-level">
      <div className="section-head">
        <p className="eyebrow">Object Context</p>
        <h2>Why Time Matters: Point vs. Track Level</h2>
        <p>
          Why do we group radar points together over time? Click the tabs below to see how adding time and motion data completely changes the ML's ability to understand the world.
        </p>
      </div>

      <div className="explainer-container">
        <div className="explainer-tabs">
          <button 
            className={`explainer-tab ${view === 'point' ? 'active' : ''}`}
            onClick={() => setView('point')}
          >
            1. Point-Level (Single Instant)
          </button>
          <button 
            className={`explainer-tab ${view === 'track' ? 'active' : ''}`}
            onClick={() => setView('track')}
          >
            2. Track-Level (Over Time)
          </button>
        </div>

        <div className="explainer-content">
          {view === 'point' && (
            <div className="explainer-pane animate-in">
              <div className="metric-box bad">
                <span className="metric-score">31%</span>
                <span className="metric-label">Accuracy</span>
              </div>
              <div className="pane-text">
                <h3>A single blip is ambiguous.</h3>
                <p>
                  At a single moment in time, a pedestrian standing still and a metal pole might reflect the exact same amount of radar energy. The ML only has limited information: range, angle, and a single velocity measurement.
                </p>
                <p><strong>Result:</strong> The ML struggles to separate classes, leading to very low accuracy and high confusion.</p>
              </div>
            </div>
          )}

          {view === 'track' && (
            <div className="explainer-pane animate-in">
              <div className="metric-box good">
                <span className="metric-score">75%</span>
                <span className="metric-label">Accuracy</span>
              </div>
              <div className="pane-text">
                <h3>Motion tells the whole story.</h3>
                <p>
                  By aggregating detections into a "track" over time, we unlock powerful new features for the ML. It can now look at the <strong>duration</strong> of the track, the <strong>density</strong> of points, and most importantly, the <strong>consistency of motion</strong>.
                </p>
                <p><strong>Result:</strong> A pedestrian's motion over time looks completely different from a pole. Accuracy more than doubles.</p>
              </div>
            </div>
          )}
        </div>

        <div className="math-details">
          <button 
            className="math-toggle" 
            onClick={() => setView(view === 'math' ? 'point' : 'math')}
          >
            {view === 'math' ? 'Hide Math Details' : 'View Exact Math & Confidence Intervals'}
          </button>
          
          {view === 'math' && (
            <div className="math-content animate-in">
              <h3>The Rigorous Math (For the Data Scientists)</h3>
              <p>
                To prove these results weren't just a lucky fluke, we ran the models many times on different data splits. 
                The numbers in the brackets <strong>[ ]</strong> represent the <strong>95% Confidence Interval</strong>. This means we are highly mathematically confident that the true real-world performance falls within that range.
              </p>
              
              <div className="math-grid">
                <div className="math-card">
                  <h4>1. Point-Level Baseline</h4>
                  <p>Model: Random Forest</p>
                  <ul>
                    <li><strong>Accuracy:</strong> 31.6% <span>[24.6% to 38.6%]</span></li>
                    <li><strong>Macro F1:</strong> 21.4% <span>[17.3% to 25.5%]</span></li>
                  </ul>
                </div>
                
                <div className="math-card highlight-card">
                  <h4>2. Track-Level Champion</h4>
                  <p>Model: XGBoost</p>
                  <ul>
                    <li><strong>Accuracy:</strong> 77.6% <span>[73.5% to 81.6%]</span></li>
                    <li><strong>Macro F1:</strong> 53.7% <span>[44.0% to 63.3%]</span></li>
                  </ul>
                </div>
              </div>
              <div className="leakage-audit" style={{ marginTop: '24px' }}>
                <strong>Leakage Audit:</strong> We strictly separated the training and testing driving sequences. The ML never saw the test tracks during training, proving that the massive jump to 77.6% accuracy is real learning, not data leakage.
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
