import { useState } from 'react';
import './DatasetAudit.css';

const stats = [
  {
    id: 1,
    value: "158",
    label: "RadarScenes sequences available",
    explanation: "The total number of driving recordings (sequences) provided in the public dataset."
  },
  {
    id: 2,
    value: "50",
    label: "sequences processed",
    explanation: "To keep compute time manageable, we processed a subset of the first 50 driving recordings."
  },
  {
    id: 3,
    value: "28.7M",
    label: "radar detections",
    explanation: "Individual radar bounces. Each point is a single reflection from an object in the environment."
  },
  {
    id: 4,
    value: "1.04M",
    label: "tracked detections",
    explanation: "Radar points that the system successfully grouped together into physical moving objects over time."
  },
  {
    id: 5,
    value: "27.7M",
    label: "untracked detections",
    explanation: "Radar points not belonging to a moving object. The vast majority of these are static environment like walls and trees."
  },
  {
    id: 6,
    value: "2,175",
    label: "track-level rows",
    explanation: "Aggregated data. Instead of looking at millions of single points, we grouped them into 2,175 distinct object summaries."
  },
  {
    id: 7,
    value: "10,971",
    label: "camera images",
    explanation: "Front-facing photos taken while driving. These are used for human reference, not for the ML model."
  },
  {
    id: 8,
    value: "438,809",
    label: "odometry samples",
    explanation: "Measurements of the car's own speed and turning rate, used to cancel out the ego-vehicle's motion."
  }
];

export default function DatasetAudit() {
  const [activeStat, setActiveStat] = useState(null);

  const toggleStat = (id) => {
    setActiveStat(activeStat === id ? null : id);
  };

  return (
    <section className="section" id="dataset">
      <div className="section-head">
        <p className="eyebrow">Dataset audit</p>
        <h2>What was processed</h2>
        <p>
          The code scans the sequence folders and processes the data. We've broken down the raw numbers below. 
          <strong> Click on any metric to see a plain-English explanation of what it means.</strong>
        </p>
      </div>
      
      <div className="stats-grid">
        {stats.map(stat => (
          <article 
            key={stat.id} 
            className={`stat-card ${activeStat === stat.id ? 'active' : ''}`}
            onClick={() => toggleStat(stat.id)}
          >
            <div className="stat-content">
              {activeStat === stat.id ? (
                <p className="stat-explanation">{stat.explanation}</p>
              ) : (
                <>
                  <span className="stat-value">{stat.value}</span>
                  <p className="stat-label">{stat.label}</p>
                </>
              )}
            </div>
            <div className="stat-hint">
              {activeStat === stat.id ? 'Click to hide' : 'Click to explain'}
            </div>
          </article>
        ))}
      </div>

      <div className="split-panel">
        <figure className="audit-figure">
          <figcaption>RadarScenes sequence structure</figcaption>
          <img src="/assets/02_sequence_structure.png" alt="RadarScenes sequence structure diagram" />
        </figure>
      </div>
    </section>
  );
}
