import { useState } from 'react';
import './SceneExplorer.css';

const scenes = {
  sceneA: {
    title: "Scene A: Sparse road / bridge approach",
    image: "/assets/04_camera_radar_timestamp_matched.png",
    sequence: "sequence_1",
    camera: "156859092964.jpg",
    dtCamera: "3.555 ms",
    dtOdometry: "0.000 ms",
    radarPoints: "408",
    labels: "400 static environment, 6 passenger car, 2 pedestrian",
    setting: "Open road / bridge approach with barriers and sparse dynamic traffic.",
    interpretation: "This frame is intentionally quiet. The important lesson is that a radar scene can be visually simple but still contain many static returns. Most points come from the built environment, while the few dynamic labels show why the classifier cannot rely only on where points appear in the plot."
  },
  sceneB: {
    title: "Scene B: Dense urban street",
    image: "/assets/14_scene_b_camera_radar_timestamp_matched.png",
    sequence: "sequence_2",
    camera: "158192068876.jpg",
    dtCamera: "3.577 ms",
    dtOdometry: "0.000 ms",
    radarPoints: "869",
    labels: "785 static environment, 57 passenger car, 20 truck, 5 large vehicle, 2 bicycle",
    setting: "Urban road with buildings, roadside structure, and several dynamic objects.",
    interpretation: "This frame is deliberately more crowded. It shows many more radar detections and more dynamic object classes. It is therefore stronger for explaining the classification challenge: the model must distinguish object types from radar features even when static infrastructure remains the majority class."
  }
};

const topics = {
  overview: {
    title: scene => `${scene.title}`,
    text: scene => `The camera frame and radar point cloud were selected by nearest timestamp. This example has ${scene.radarPoints} radar detections near the frame, with labels: ${scene.labels}. The radar plot is in coordinate space, not projected into the camera image.`
  },
  camera: {
    title: () => "Camera frame",
    text: scene => `The camera image documents what the road looked like in ${scene.sequence}. It helps a human interpret the scene setting (${scene.setting}), but the camera pixels are not used as supervised inputs for the classifier in this project.`
  },
  radar: {
    title: () => "Radar detections",
    text: scene => `The radar side is the actual classification data. In this frame neighborhood there are ${scene.radarPoints} detections. Each point has range, azimuth, RCS, radial velocity, compensated radial velocity, coordinates, sensor_id, track_id, and the ground-truth label_id.`
  },
  time: {
    title: () => "Nearest timestamp",
    text: scene => `The camera file ${scene.camera} is matched to the nearest radar timestamp for ${scene.sequence}. The recorded radar-camera difference is ${scene.dtCamera}; this is why the figure is called timestamp-aligned rather than synchronized or projected.`
  },
  odometry: {
    title: () => "Odometry link",
    text: scene => `Odometry is linked by nearest timestamp as well. For this selected radar timestamp the nearest odometry sample is effectively aligned (${scene.dtOdometry} in the generated link table), giving ego-motion context for car coordinates, sequence coordinates, and compensated velocity.`
  }
};

export default function SceneExplorer() {
  const [activeSceneId, setActiveSceneId] = useState('sceneA');
  const [activeTopic, setActiveTopic] = useState('overview');
  
  const scene = scenes[activeSceneId];
  const topic = topics[activeTopic];

  return (
    <section className="section" id="scene-explorer">
      <div className="section-head">
        <p className="eyebrow">Camera, radar, odometry</p>
        <h2>Interactive Data Context</h2>
      </div>
      
      <div className="scene-tabs">
        <button 
          className={`tab-btn ${activeSceneId === 'sceneA' ? 'active' : ''}`}
          onClick={() => { setActiveSceneId('sceneA'); setActiveTopic('overview'); }}
        >
          Scene A
        </button>
        <button 
          className={`tab-btn ${activeSceneId === 'sceneB' ? 'active' : ''}`}
          onClick={() => { setActiveSceneId('sceneB'); setActiveTopic('overview'); }}
        >
          Scene B
        </button>
      </div>

      <div className="relationship-grid">
        <div className="relationship-visual">
          <img src={scene.image} alt={scene.title} className="relationship-image" />
          
          <button 
            className={`hotspot h-camera ${activeTopic === 'camera' ? 'active' : ''}`} 
            onClick={() => setActiveTopic('camera')}
          >
            Camera frame
          </button>
          
          <button 
            className={`hotspot h-radar ${activeTopic === 'radar' ? 'active' : ''}`} 
            onClick={() => setActiveTopic('radar')}
          >
            Radar detections
          </button>
          
          <button 
            className={`hotspot h-time ${activeTopic === 'time' ? 'active' : ''}`} 
            onClick={() => setActiveTopic('time')}
          >
            Nearest timestamp
          </button>
          
          <button 
            className={`hotspot h-odom ${activeTopic === 'odometry' ? 'active' : ''}`} 
            onClick={() => setActiveTopic('odometry')}
          >
            Odometry link
          </button>
        </div>

        <article className="relationship-card">
          <h3>{topic.title(scene)}</h3>
          <p>{topic.text(scene)}</p>
        </article>
      </div>

      <div className="scene-meta-grid">
        <article className="meta-card">
          <h3>Selected frame metadata</h3>
          <dl className="metadata-list">
            <dt>Sequence</dt>
            <dd>{scene.sequence}</dd>
            <dt>Camera frame</dt>
            <dd>{scene.camera}</dd>
            <dt>Radar-camera offset</dt>
            <dd>{scene.dtCamera}</dd>
            <dt>Nearby radar detections</dt>
            <dd>{scene.radarPoints}</dd>
            <dt>Label distribution</dt>
            <dd>{scene.labels}</dd>
          </dl>
        </article>
        <article className="meta-card">
          <h3>What this scene teaches</h3>
          <p className="interpretation">{scene.interpretation}</p>
        </article>
      </div>

      <article className="comparison-card">
        <h3>Difference between the two scenes</h3>
        <div className="comparison-columns">
          <div>
            <h4>Scene A: sparse road / bridge approach</h4>
            <p>
              The radar neighborhood is dominated by static environment detections. Only a few passenger-car and pedestrian labels appear, so the frame is useful for explaining that most radar returns can come from road boundaries, barriers, poles, vegetation, and other non-moving structure.
            </p>
          </div>
          <div>
            <h4>Scene B: dense urban street</h4>
            <p>
              The radar neighborhood contains more than twice as many points and several dynamic classes: passenger cars, truck, large vehicle, and bicycles. This is a better example of why class imbalance and multi-class separation matter in the final ML task.
            </p>
          </div>
          <div>
            <h4>What stays the same</h4>
            <p>
              In both examples the camera explains the road scene visually, the radar point cloud is the supervised data source, labels come from <code>label_id</code>, and odometry links car-frame coordinates to sequence-level motion. The image is not a calibrated projection; it is an auditable timestamp-aligned comparison.
            </p>
          </div>
        </div>
      </article>
    </section>
  );
}
