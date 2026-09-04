import React from 'react';
import { motion } from 'framer-motion';

export default function FTCA() {
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">FTCA Architecture</h2>
          <p className="text-gray-400 text-lg">Frequency-Temporal Cross-Attention</p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-16 items-start">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex flex-col items-center font-mono text-sm w-full"
          >
            <div className="w-full max-w-sm border border-gray-700 bg-gray-900 p-4 text-center text-gray-300">
              Input: Video [B, 3, 16, 224, 224]
            </div>
            
            <div className="my-2 text-gray-600">↓</div>
            
            <div className="w-full max-w-sm border border-gray-700 bg-gray-900 p-4 text-center">
              <div className="text-white mb-1">R3D-18 Backbone</div>
              <div className="text-xs text-gray-500 mb-2">(pretrained on Kinetics-400)</div>
              <div className="text-gray-400">→ Temporal features [B, 512]</div>
            </div>
            
            <div className="my-2 text-gray-600">↓</div>
            
            <div className="w-full max-w-sm border border-gray-700 bg-gray-900 p-4 text-center">
              <div className="text-white mb-1">FrequencyEncoder</div>
              <div className="text-xs text-gray-500 mb-2">(torch.fft.fft2)</div>
              <div className="text-gray-400">→ Frequency features [B, 512]</div>
            </div>

            <div className="my-2 text-gray-600">↓</div>
            
            <div className="w-full max-w-sm border border-gray-700 bg-gray-900 p-4 text-center">
              <div className="text-white mb-2">Cross-Attention (8 heads)</div>
              <div className="text-gray-400">→ Fused features [B, 512]</div>
            </div>

            <div className="my-2 text-gray-600">↓</div>
            
            <div className="w-full max-w-sm border border-gray-700 bg-gray-900 p-4 text-center">
              <div className="text-white mb-1">Classification Head</div>
              <div className="text-xs text-gray-500 mb-2">(Linear 512→1)</div>
              <div className="text-gray-400">→ P(deepfake)</div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="w-full"
          >
            <h3 className="text-xl font-bold mb-6 text-white border-b border-gray-800 pb-2">Training Specs</h3>
            
            <table className="w-full text-sm text-left text-gray-400 font-mono">
              <tbody>
                <tr className="border-b border-gray-800">
                  <td className="py-3 pr-4 text-gray-500">Dataset</td>
                  <td className="py-3 text-white">FF++ C23 (7k) + Celeb-DF v2 (6k)</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 pr-4 text-gray-500">GPU Compute</td>
                  <td className="py-3 text-white">NVIDIA A40 48GB (RunPod)</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 pr-4 text-gray-500">Phase 2</td>
                  <td className="py-3 text-white">18 epochs, early stop at 11</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 pr-4 text-gray-500">Phase 3 (Domain Adapt)</td>
                  <td className="py-3 text-white">8 epochs</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 pr-4 text-gray-500">Optimizer</td>
                  <td className="py-3 text-white">AdamW, LR 1e-4, Cosine Annealing</td>
                </tr>
                <tr>
                  <td className="py-3 pr-4 text-gray-500">Augmentation</td>
                  <td className="py-3 text-white">Random horiz flip, Label smooth 0.05</td>
                </tr>
              </tbody>
            </table>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
