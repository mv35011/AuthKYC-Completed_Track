import React from 'react';
import { motion } from 'framer-motion';

export default function Results() {
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Model Performance</h2>
          <p className="text-gray-400 text-lg">Evaluation on validation sets</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mb-16 overflow-x-auto"
        >
          <table className="w-full text-left border-collapse min-w-[600px]">
            <thead>
              <tr className="border-b border-gray-800 text-sm font-mono text-gray-500 uppercase tracking-wider">
                <th className="py-4 px-4 font-normal">Metric / Phase</th>
                <th className="py-4 px-4 font-normal">Phase 2 (Main)</th>
                <th className="py-4 px-4 font-normal">Phase 3 (Domain Adapt)</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              <tr className="border-b border-gray-800/50">
                <td className="py-4 px-4 text-gray-400 font-mono">Dataset</td>
                <td className="py-4 px-4 text-white">FF++ C23 + Celeb-DF v2</td>
                <td className="py-4 px-4 text-white">Celeb-DF v2</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-4 px-4 text-gray-400 font-mono">Best Epoch</td>
                <td className="py-4 px-4 text-white">11/18</td>
                <td className="py-4 px-4 text-white">3/8</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-4 px-4 text-gray-400 font-mono">Validation Accuracy</td>
                <td className="py-4 px-4 text-white">91.74%</td>
                <td className="py-4 px-4 text-emerald-400">95.99%</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-4 px-4 text-gray-400 font-mono">Validation Loss</td>
                <td className="py-4 px-4 text-white">0.2194</td>
                <td className="py-4 px-4 text-white">0.1331</td>
              </tr>
              <tr>
                <td className="py-6 px-4 text-gray-400 font-mono">Validation AUC</td>
                <td className="py-6 px-4 text-3xl font-bold text-white">97.34%</td>
                <td className="py-6 px-4 text-3xl font-bold text-emerald-500">99.00%</td>
              </tr>
            </tbody>
          </table>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="border border-gray-800 p-6 flex flex-col md:flex-row justify-between text-sm font-mono text-gray-400"
        >
          <div className="mb-4 md:mb-0">
            <span className="text-gray-500 block text-xs uppercase mb-1">Training Time</span>
            <span className="text-white">~6 hours</span>
          </div>
          <div className="mb-4 md:mb-0">
            <span className="text-gray-500 block text-xs uppercase mb-1">Framework</span>
            <span className="text-white">PyTorch 2.4.1 + CUDA 12.4</span>
          </div>
          <div>
            <span className="text-gray-500 block text-xs uppercase mb-1">Inference Target</span>
            <span className="text-white">HF ZeroGPU (A10G)</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
