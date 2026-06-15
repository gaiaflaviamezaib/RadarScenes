import { useState } from 'react';
import Hero from './components/Hero';
import SimpleDataExplanation from './components/SimpleDataExplanation';
import DatasetAudit from './components/DatasetAudit';
import FaqAccordion from './components/FaqAccordion';
import VisualEvidence from './components/VisualEvidence';
import SceneExplorer from './components/SceneExplorer';
import TrackLevelExplainer from './components/TrackLevelExplainer';
import ModelComparison from './components/ModelComparison';
import Plot3D from './components/Plot3D';
import PipelineRunner from './components/PipelineRunner';

function App() {
  const [activeTab, setActiveTab] = useState('intro');

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setTimeout(() => {
      const content = document.querySelector('.tab-content');
      if (content) {
        const y = content.getBoundingClientRect().top + window.scrollY - 100;
        window.scrollTo({ top: y, behavior: 'smooth' });
      }
    }, 10);
  };

  return (
    <div className="container">
      <Hero onExplore3D={() => handleTabChange('3d')} />
      
      <nav className="main-tabs">
        <button 
          className={`tab-link ${activeTab === 'dataset' ? 'active' : ''}`}
          onClick={() => handleTabChange('dataset')}
        >
          1. Dataset Overview
        </button>
        <button 
          className={`tab-link ${activeTab === 'scenes' ? 'active' : ''}`}
          onClick={() => handleTabChange('scenes')}
        >
          2. Scene Analysis
        </button>
        <button 
          className={`tab-link ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => handleTabChange('models')}
        >
          3. ML & Models
        </button>
        <button 
          className={`tab-link ${activeTab === '3d' ? 'active' : ''}`}
          onClick={() => handleTabChange('3d')}
        >
          4. Interactive 3D
        </button>
        <button 
          className={`tab-link ${activeTab === 'pipeline' ? 'active' : ''}`}
          onClick={() => handleTabChange('pipeline')}
        >
          5. Run Pipeline
        </button>
        <button 
          className={`tab-link ${activeTab === 'faq' ? 'active' : ''}`}
          onClick={() => handleTabChange('faq')}
        >
          6. FAQs
        </button>
      </nav>

      <div className="tab-content animate-in">
        {activeTab === 'dataset' && (
          <>
            <SimpleDataExplanation />
            <DatasetAudit />
            <VisualEvidence />
          </>
        )}

        {activeTab === 'scenes' && (
          <SceneExplorer />
        )}

        {activeTab === 'models' && (
          <>
            <ModelComparison />
            <TrackLevelExplainer />
          </>
        )}

        {activeTab === '3d' && (
          <Plot3D />
        )}

        {activeTab === 'pipeline' && (
          <PipelineRunner />
        )}

        {activeTab === 'faq' && (
          <FaqAccordion />
        )}
      </div>
    </div>
  );
}

export default App;
