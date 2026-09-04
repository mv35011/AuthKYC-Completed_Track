import React from 'react';
import { motion } from 'framer-motion';

const Credits = () => {
  return (
    <section className="min-h-screen bg-[#0a0a1a] flex flex-col justify-center items-center px-6 py-20 text-center">
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="max-w-3xl w-full"
      >
        <h2 className="text-2xl text-gray-500 mb-8">Built by</h2>
        
        <div className="mb-10">
          <h3 className="text-3xl font-bold text-white mb-2">Manmohan Vishwakarma</h3>
          <p className="text-gray-400 font-mono text-sm">NIT Patna (NITP), ECE 2027</p>
        </div>

        <div className="mb-12">
          <div className="text-xs uppercase tracking-widest text-gray-600 mb-4 font-semibold">Tech Stack</div>
          <div className="flex flex-wrap justify-center gap-3">
            {['PyTorch', 'HuggingFace', 'React', 'Vercel', 'MediaPipe', 'ZeroGPU'].map((tech) => (
              <span key={tech} className="px-3 py-1 bg-gray-900 border border-gray-800 text-sm text-gray-300 font-mono">
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <a
            href="https://huggingface.co/spaces/mv350113/authkyc-demo"
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-2 bg-white text-black font-medium hover:bg-gray-200 transition-colors text-sm"
          >
            Try the Demo
          </a>
          <a
            href="https://github.com/mv35011/AuthKYC-Completed_Track"
            target="_blank"
            rel="noopener noreferrer"
            className="px-6 py-2 border border-gray-600 text-white font-medium hover:bg-gray-900 transition-colors text-sm"
          >
            GitHub Repo
          </a>
        </div>

        <div className="text-xs font-mono text-gray-600">
          Built for Razorpay FTX Buildathon 2025
        </div>
      </motion.div>
    </section>
  );
};

export default Credits;
