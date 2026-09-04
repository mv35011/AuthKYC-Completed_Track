import React from 'react';
import { motion } from 'framer-motion';

const Hero = () => {
  return (
    <section className="min-h-screen bg-[#0a0a1a] flex flex-col justify-center items-center text-center px-6 py-20">
      <div className="max-w-5xl mx-auto flex flex-col items-center">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="mb-8 px-4 py-2 border border-gray-800 text-gray-400 text-sm font-mono uppercase tracking-wide"
        >
          Razorpay FTX Buildathon 2025 — AI Risk Manager Track
        </motion.div>

        <motion.h1
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-5xl md:text-7xl font-bold mb-6 tracking-tight text-white"
        >
          <span className="text-purple-400">AuthKYC</span>
        </motion.h1>

        <motion.h2
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-2xl md:text-3xl text-gray-300 font-medium mb-8 max-w-3xl"
        >
          Defending Bank KYC Against AI Attacks
        </motion.h2>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-lg text-gray-400 max-w-2xl leading-relaxed mb-16 mx-auto"
        >
          An end-to-end Presentation Attack Detection pipeline that protects bank KYC video verification from virtual camera injection, screen replay attacks, and AI-generated deepfakes.
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mt-12 text-gray-500 font-mono text-xl"
        >
          ↓
        </motion.div>
      </div>
    </section>
  );
};

export default Hero;
