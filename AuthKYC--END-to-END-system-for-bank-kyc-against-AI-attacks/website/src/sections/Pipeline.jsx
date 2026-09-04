import React from 'react';
import { motion } from 'framer-motion';

const stages = [
  {
    num: '01',
    title: 'PRNU Sensor Forensics',
    color: 'border-cyan-500',
    desc: 'Extracts camera sensor fingerprint.',
    detects: 'Virtual cameras (OBS, ManyCam)'
  },
  {
    num: '02',
    title: 'Moiré FFT Analysis',
    color: 'border-purple-500',
    desc: 'Analyzes frequency spectrum via 2D FFT.',
    detects: 'Screen replay attacks'
  },
  {
    num: '03',
    title: 'rPPG Pulse Extraction',
    color: 'border-emerald-500',
    desc: 'Extracts remote photoplethysmography signal.',
    detects: 'Non-biological liveness'
  },
  {
    num: '04',
    title: 'FTCA Deepfake Detection',
    color: 'border-orange-500',
    desc: 'Frequency-Temporal Cross-Attention.',
    detects: 'AI-generated deepfakes'
  }
];

export default function Pipeline() {
  return (
    <section className="min-h-screen bg-[#0a0a1a] py-20 px-6 flex flex-col justify-center items-center">
      <div className="max-w-5xl w-full">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Pipeline Architecture</h2>
          <p className="text-gray-400 text-lg">4-Stage Defense-in-Depth</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex flex-col lg:flex-row gap-4 items-center justify-between"
        >
          {stages.map((stage, index) => (
            <React.Fragment key={index}>
              <div className={`flex-1 bg-gray-900 border border-gray-800 border-l-4 ${stage.color} p-5 w-full max-w-sm`}>
                <div className="text-xs text-gray-500 font-mono mb-1">Stage {stage.num}</div>
                <h3 className="text-base font-semibold text-white mb-2">{stage.title}</h3>
                <p className="text-sm text-gray-400 mb-3">{stage.desc}</p>
                <div className="text-xs bg-gray-800 p-2 text-gray-300">
                  <span className="font-semibold text-gray-400">Detects:</span> {stage.detects}
                </div>
              </div>
              
              {index < stages.length - 1 && (
                <div className="text-gray-600 font-mono text-xl lg:rotate-0 rotate-90 my-2 lg:my-0">
                  →
                </div>
              )}
            </React.Fragment>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <div className="inline-block px-6 py-3 border border-gray-800 text-gray-300 font-mono text-sm uppercase tracking-wider">
            Waterfall architecture: if any stage fails → KYC denied
          </div>
        </motion.div>
      </div>
    </section>
  );
}
