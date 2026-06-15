import { useState } from 'react';
import './VisualEvidence.css';

const figuresData = [
  { category: "data",     title: "Dataset hierarchy",              src: "/assets/01_dataset_hierarchy.png",                       caption: "Project organization from raw sequences to outputs." },
  { category: "data",     title: "Sequence & Detection Hierarchy", src: "/assets/edu_sequence_detection_hierarchy.png",           caption: "A visual breakdown of the dataset structure, explaining how continuous driving sequences are divided into individual radar scenes and tracked object IDs." },
  { category: "data",     title: "Asynchronous Sensor Stream",     src: "/assets/edu_sensor_stream.png",                          caption: "Demonstrates how independent, asynchronous radar measurements from four different vehicle sensors are merged into a single chronological timeline." },
  { category: "data",     title: "Radar point cloud",              src: "/assets/03_radar_point_cloud_by_type.png",               caption: "Detections in car coordinates colored by object type." },
  { category: "data",     title: "Camera and radar alignment A",   src: "/assets/04_camera_radar_timestamp_matched.png",          caption: "Sparse road example with 408 nearby radar detections." },
  { category: "data",     title: "Camera and radar alignment B",   src: "/assets/14_scene_b_camera_radar_timestamp_matched.png",  caption: "Dense urban example with 869 nearby radar detections and multiple dynamic classes." },
  { category: "data",     title: "Duration per sequence",          src: "/assets/duration_per_sequence.png",                      caption: "Sequence durations are unequal, which is expected." },
  { category: "data",     title: "Radar points per sequence",      src: "/assets/radar_points_per_sequence.png",                  caption: "Sample contribution varies naturally by sequence length and driving content." },
  { category: "data",     title: "Camera images per sequence",     src: "/assets/camera_images_per_sequence.png",                 caption: "Camera availability for timestamp-aligned context." },
  { category: "features", title: "RCS vs range",                   src: "/assets/05_rcs_vs_range.png",                            caption: "Reflectivity feature compared against range by object type." },
  { category: "features", title: "Velocity vs range",              src: "/assets/06_velocity_vs_range.png",                       caption: "Compensated radial velocity compared against range." },
  { category: "features", title: "3D feature-space view",          src: "/assets/13_3d_feature_space.png",                        caption: "Static export of the 3D feature-space view." },
  { category: "models",   title: "Random Forest Majority Voting",  src: "/assets/edu_random_forest_voting.png",                   caption: "An illustration of the ensemble learning process used in the project, where multiple decision trees cast a vote to determine the final semantic class." },
  { category: "workflow", title: "Pipeline workflow",              src: "/assets/12_workflow.png",                                caption: "Raw sequences to feature extraction, linking, models, and evaluation." },
  { category: "workflow", title: "End-to-End ML Pipeline",         src: "/assets/edu_ml_pipeline.png",                            caption: "The complete 7-step machine learning workflow, from ingesting raw Kaggle sequences to training the models and extracting feature importance." },
];

export default function VisualEvidence() {
  const [filter, setFilter] = useState('all');
  const [modalImage, setModalImage] = useState(null);

  const filteredFigures = filter === 'all' 
    ? figuresData 
    : figuresData.filter(fig => fig.category === filter);

  return (
    <section className="section" id="visuals">
      <div className="section-head">
        <p className="eyebrow">Visual Evidence</p>
        <h2>Final visual elements</h2>
        <p>
          All categorical radar labels and sensor IDs use discrete colors and legends. Use the filter buttons to browse the generated figures.
        </p>
      </div>

      <div className="filter-row">
        <button className={`chip ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>All</button>
        <button className={`chip ${filter === 'data' ? 'active' : ''}`} onClick={() => setFilter('data')}>Data</button>
        <button className={`chip ${filter === 'features' ? 'active' : ''}`} onClick={() => setFilter('features')}>Features</button>
        <button className={`chip ${filter === 'models' ? 'active' : ''}`} onClick={() => setFilter('models')}>Models</button>
        <button className={`chip ${filter === 'workflow' ? 'active' : ''}`} onClick={() => setFilter('workflow')}>Workflow</button>
      </div>

      <div className="figure-grid">
        {filteredFigures.map((fig, index) => (
          <article 
            key={index} 
            className="figure-card" 
            onClick={() => setModalImage(fig)}
          >
            <img src={fig.src} alt={fig.title} />
            <div className="figure-content">
              <h3>{fig.title}</h3>
              <p>{fig.caption}</p>
            </div>
          </article>
        ))}
      </div>

      {modalImage && (
        <div className="modal open" onClick={() => setModalImage(null)}>
          <button className="modal-close" onClick={() => setModalImage(null)}>×</button>
          <img 
            src={modalImage.src} 
            alt={modalImage.title} 
            onClick={(e) => e.stopPropagation()} 
          />
          <p className="modal-caption">{modalImage.title}: {modalImage.caption}</p>
        </div>
      )}
    </section>
  );
}
