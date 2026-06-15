
import './Hero.css';

export default function Hero({ onExplore3D }) {
  return (
    <section className="hero">
      <div className="hero-content">
        <p className="eyebrow">RadarScenes Project</p>
        <h1>Decoding the built environment through radar.</h1>
        <p className="subtitle">
          An exploration of the RadarScenes dataset: linking automotive radar point clouds to driving context, understanding object signatures in feature space, and evaluating machine learning classifiers.
        </p>
        <div className="hero-actions">
          <button className="button primary" onClick={onExplore3D}>Explore 3D View</button>
          <a 
            className="button secondary" 
            href="https://www.kaggle.com/datasets/aleksandrdubrovin/the-radarscenes-data-set" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            View Dataset on Kaggle
          </a>
          <a 
            className="button secondary" 
            href="/RadarScenes_Semantic_Prediction_Report.pdf" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Read Paper
          </a>
        </div>
        <div className="hero-image-wrapper">
          <img 
            src="/assets/simple_radar_diagram.png" 
            alt="Top-down diagram of a car with radar cones" 
            className="hero-image animate-in"
          />
        </div>
      </div>
    </section>
  );
}
