import React, { useState, useEffect } from 'react';
import Hero from './sections/Hero';
import Problem from './sections/Problem';
import Pipeline from './sections/Pipeline';
import FTCA from './sections/FTCA';
import Results from './sections/Results';
import Demo from './sections/Demo';
import Shortcomings from './sections/Shortcomings';
import Credits from './sections/Credits';

const sections = [
  { id: 'hero', label: 'Hero' },
  { id: 'problem', label: 'Problem' },
  { id: 'pipeline', label: 'Pipeline' },
  { id: 'ftca', label: 'FTCA' },
  { id: 'results', label: 'Results' },
  { id: 'demo', label: 'Demo' },
  { id: 'shortcomings', label: 'Limitations' },
  { id: 'credits', label: 'Credits' },
];

function App() {
  const [activeSection, setActiveSection] = useState('hero');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.3 }
    );

    sections.forEach(({ id }) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  const scrollTo = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="relative w-full bg-[#0a0a1a] text-white">
      {/* Side Navigation Dots */}
      <div className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col gap-3">
        {sections.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => scrollTo(id)}
            className="group relative flex items-center justify-end"
            aria-label={`Scroll to ${label}`}
          >
            <span
              className="absolute right-8 px-2 py-1 rounded bg-gray-800 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none"
            >
              {label}
            </span>
            <div
              className={`w-3 h-3 rounded-full transition-all duration-300 ${
                activeSection === id
                  ? 'bg-purple-500 scale-125'
                  : 'bg-gray-600 hover:bg-gray-400'
              }`}
            />
          </button>
        ))}
      </div>

      <main>
        <section id="hero"><Hero /></section>
        <section id="problem"><Problem /></section>
        <section id="pipeline"><Pipeline /></section>
        <section id="ftca"><FTCA /></section>
        <section id="results"><Results /></section>
        <section id="demo"><Demo /></section>
        <section id="shortcomings"><Shortcomings /></section>
        <section id="credits"><Credits /></section>
      </main>
    </div>
  );
}

export default App;
