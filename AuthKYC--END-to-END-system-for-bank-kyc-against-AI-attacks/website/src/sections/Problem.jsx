import React from 'react';
import { motion } from 'framer-motion';

const problems = [
  {
    icon: '🎥',
    title: 'Virtual Camera Injection',
    description: 'Attackers use OBS, ManyCam to inject pre-recorded or synthetic video feeds into the KYC process.',
    borderColor: 'border-blue-500'
  },
  {
    icon: '📺',
    title: 'Screen Replay Attack',
    description: "Holding a phone in front of a screen playing a victim's genuine KYC video.",
    borderColor: 'border-purple-500'
  },
  {
    icon: '🤖',
    title: 'AI Deepfakes',
    description: 'Face-swapped or fully generated videos using DeepFakes, Face2Face, FaceShifter.',
    borderColor: 'border-red-500'
  },
  {
    icon: '🧟',
    title: 'Non-Biological Sources',
    description: 'Photos, masks, 3D models held up to the camera.',
    borderColor: 'border-amber-500'
  }
];

const Problem = () => {
  return (
    <section className="min-h-screen bg-[#0a0a1a] flex flex-col justify-center px-6 py-20">
      <div className="max-w-5xl mx-auto w-full">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">The Problem</h2>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Current KYC systems are vulnerable to four primary attack vectors.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full"
        >
          {problems.map((problem, index) => (
            <div
              key={index}
              className={`bg-gray-900 border border-gray-800 border-l-4 ${problem.borderColor} p-6 flex flex-col`}
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xl">{problem.icon}</span>
                <h3 className="text-lg font-semibold text-white">
                  {problem.title}
                </h3>
              </div>
              <p className="text-gray-400 text-sm leading-relaxed">
                {problem.description}
              </p>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Problem;
