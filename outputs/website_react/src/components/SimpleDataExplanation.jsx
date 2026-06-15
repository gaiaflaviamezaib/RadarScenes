
import './SimpleDataExplanation.css';

export default function SimpleDataExplanation() {
  return (
    <section className="section" id="data-101">
      <div className="section-head">
        <p className="eyebrow">Data 101</p>
        <h2>What is this data?</h2>
      </div>
      <div className="explanation-grid">
        <article className="explanation-card">
          <h3>The Setup</h3>
          <p>
            A car equipped with four radar sensors and a camera drove around various environments, ranging from sparse highways to dense urban streets.
          </p>
        </article>
        <article className="explanation-card">
          <h3>The Radar Data</h3>
          <p>
            Unlike a camera that captures pixels, radar sends out radio waves and listens for them to bounce back. The dataset consists of thousands of these "bounces" (or detections) representing physical objects.
          </p>
        </article>
        <article className="explanation-card">
          <h3>The Goal</h3>
          <p>
            We built a Machine Learning model that relies purely on radar data to automatically classify objects like cars, pedestrians, or static walls. It makes these predictions by analyzing specific measurements, including the distance of the radar bounce and the speed of the object.
          </p>
        </article>
      </div>
    </section>
  );
}
