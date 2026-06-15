import { useEffect, useRef, useState } from 'react';
import './Plot3D.css';

const labelColors = {
  0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728",
  4: "#9467bd", 5: "#8c564b", 6: "#e377c2", 7: "#7f7f7f",
  8: "#bcbd22", 9: "#17becf", 10: "#aec7e8", 11: "#ffbb78"
};

const labelNames = {
  0: "passenger car", 1: "large vehicle", 2: "truck", 3: "bus",
  4: "train", 5: "bicycle", 6: "motorized two-wheeler", 7: "pedestrian",
  8: "pedestrian group", 9: "animal", 10: "other dynamic object", 11: "static environment"
};

export default function Plot3D() {
  const plotRef = useRef(null);
  const [data] = useState(() => window.RADAR_SAMPLE || []);
  
  const [xAxis, setXAxis] = useState('x_cc');
  const [yAxis, setYAxis] = useState('y_cc');
  const [zAxis, setZAxis] = useState('vr_compensated');
  const [pointLimit, setPointLimit] = useState(5000);

  useEffect(() => {
    if (!data.length || !plotRef.current || !window.Plotly) return;

    let filtered = data.slice(0, pointLimit);

    const traces = [...new Set(filtered.map(row => row.label_id))]
      .sort((a, b) => a - b)
      .map(id => {
        const group = filtered.filter(row => row.label_id === id);
        return {
          type: "scatter3d",
          mode: "markers",
          name: `${id}: ${labelNames[id] || "unknown"}`,
          x: group.map(row => row[xAxis]),
          y: group.map(row => row[yAxis]),
          z: group.map(row => row[zAxis]),
          text: group.map(row => `label: ${row.label_name}<br>sequence: ${row.sequence_id}`),
          hovertemplate: `%{text}<br>${xAxis}: %{x:.3f}<br>${yAxis}: %{y:.3f}<br>${zAxis}: %{z:.3f}<extra></extra>`,
          marker: {
            size: 3,
            color: labelColors[id] || "#333333",
            opacity: 0.8
          }
        };
      });

    const layout = {
      title: {
        text: `${xAxis} vs ${yAxis} vs ${zAxis}<br><sup>${filtered.length.toLocaleString()} sampled detections</sup>`,
        x: 0.02,
        xanchor: "left",
        font: { color: "#1E293B" }
      },
      scene: {
        xaxis: { title: xAxis, gridcolor: "#E5E7EB", backgroundcolor: "#F3F4F6", showbackground: true },
        yaxis: { title: yAxis, gridcolor: "#E5E7EB", backgroundcolor: "#F3F4F6", showbackground: true },
        zaxis: { title: zAxis, gridcolor: "#E5E7EB", backgroundcolor: "#F3F4F6", showbackground: true },
        camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } }
      },
      legend: { font: { color: "#1E293B" } },
      margin: { l: 0, r: 0, t: 60, b: 0 },
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent"
    };

    const currentPlot = plotRef.current;
    window.Plotly.react(currentPlot, traces, layout, { responsive: true, displaylogo: false });

    // Cleanup Plotly instance on unmount to prevent memory leaks
    return () => {
      if (currentPlot && window.Plotly) {
        window.Plotly.purge(currentPlot);
      }
    };
  }, [data, xAxis, yAxis, zAxis, pointLimit]);

  const numericColumns = data.length ? Object.keys(data[0]).filter(key => typeof data[0][key] === "number") : [];

  return (
    <section className="section" id="interactive-3d">
      <div className="section-head">
        <p className="eyebrow">Interactive 3D</p>
        <h2>Explore the Feature Space</h2>
      </div>
      
      <div className="plot-container">
        <div className="controls">
          <div className="control-group">
            <label>X-Axis</label>
            <select value={xAxis} onChange={e => setXAxis(e.target.value)}>
              {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label>Y-Axis</label>
            <select value={yAxis} onChange={e => setYAxis(e.target.value)}>
              {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label>Z-Axis</label>
            <select value={zAxis} onChange={e => setZAxis(e.target.value)}>
              {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
            </select>
          </div>
          <div className="control-group">
            <label>Points: {pointLimit}</label>
            <input 
              type="range" 
              min="1000" max="25000" step="1000" 
              value={pointLimit} 
              onChange={e => setPointLimit(Number(e.target.value))} 
            />
          </div>
        </div>
        
        <div className="plot-stage" ref={plotRef}></div>
      </div>
    </section>
  );
}
