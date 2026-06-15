import { useState } from 'react';
import './FaqAccordion.css';

const faqs = [
  {
    question: "How was the RadarScenes dataset recorded?",
    answer: "RadarScenes was recorded using a measurement vehicle equipped with four automotive radar sensors. It also includes odometry data and a front-facing documentary camera for contextual scene understanding."
  },
  {
    question: "What exactly is a radar detection in this dataset?",
    answer: "Each row in the dataset represents a single radar detection. This includes the timestamp, the specific sensor ID, the object's range and azimuth (angle), the radar cross-section (RCS), and radial velocity."
  },
  {
    question: "What is the ratio of tracked to untracked objects?",
    answer: "In our 50-sequence analysis, we found about 1 million detections with tracked IDs (moving objects) compared to 27.7 million untracked detections, which mostly represent the static environment like buildings and roads."
  },
  {
    question: "How is the dataset split for training and testing?",
    answer: "While RadarScenes provides official tags, we implemented a custom sequence-aware split. This ensures that entire driving sequences are kept intact within either the train or validation sets to prevent data leakage."
  },
  {
    question: "How long is a typical driving sequence?",
    answer: "In the 50 processed sequences, the driving durations range from 13 seconds up to 3.5 minutes, with the average sequence lasting about 84 seconds."
  },
  {
    question: "What is the layout of the four radar sensors?",
    answer: "The four automotive radar sensors are mounted around the vehicle to provide 360-degree coverage. The 'sensor_id' field tracks which specific radar produced the detection, compensating for mounting angle and yaw."
  }
];

export default function FaqAccordion() {
  const [openIndex, setOpenIndex] = useState(null);

  const toggleOpen = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="section" id="faq">
      <div className="section-head">
        <p className="eyebrow">Data Guide</p>
        <h2>Frequently Asked Questions</h2>
        <p>
          Quick answers based on the RadarScenes documentation and the outputs generated from the 50 processed sequences.
        </p>
      </div>

      <div className="accordion-container">
        {faqs.map((faq, index) => {
          const isOpen = openIndex === index;
          return (
            <div 
              key={index} 
              className={`accordion-item ${isOpen ? 'open' : ''}`}
            >
              <button 
                className="accordion-header" 
                onClick={() => toggleOpen(index)}
                aria-expanded={isOpen}
              >
                <span className="accordion-title">{faq.question}</span>
                <span className="accordion-icon">{isOpen ? '−' : '+'}</span>
              </button>
              <div 
                className="accordion-content" 
                style={{ maxHeight: isOpen ? (faq.image ? '600px' : '200px') : '0' }}
              >
                <div className="accordion-content-inner">
                  <p>{faq.answer}</p>
                  {faq.image && (
                    <img src={faq.image} alt={faq.question} className="faq-image" />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
