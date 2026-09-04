import React from 'react';
import { motion } from 'framer-motion';

const limitations = [
  {
    title: 'Domain Gap',
    desc: 'Model trained on FF++/CelebDF datasets performs differently on raw phone camera selfies. Videos uploaded through web services get re-encoded, destroying PRNU sensor fingerprints. Domain adaptation with device-specific data is needed.'
  },
  {
    title: 'ONNX Export',
    desc: 'torch.fft.fft2 (used in FrequencyEncoder) has no ONNX equivalent. Cannot export to ONNX for edge deployment. Using PyTorch directly for inference.'
  },
  {
    title: 'PRNU Re-encoding',
    desc: 'Web platforms (HF Spaces, WhatsApp, etc.) re-encode uploaded videos, destroying the camera sensor noise pattern. PRNU works best on raw, uncompressed video from the original device.'
  },
  {
    title: 'Real-time Latency',
    desc: 'Full 4-stage pipeline takes ~15-17 seconds per video on ZeroGPU A10G. For real-time KYC, stages could be parallelized or the model distilled.'
  }
];

const futures = [
  'Few-shot domain adaptation for specific phone models',
  'Model distillation for edge deployment (mobile SDK)',
  'Integration with Aadhaar eKYC / DigiLocker APIs',
  'Adversarial training against emerging attack methods'
];

export default function Shortcomings() {
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4 text-white">Limitations & Future Work</h2>
          <p className="text-gray-400 text-lg">System constraints and next steps</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid md:grid-cols-2 gap-6 mb-16"
        >
          {limitations.map((lim, i) => (
            <div
              key={i}
              className="bg-gray-900 border border-gray-800 border-l-4 border-l-amber-500 p-6"
            >
              <h3 className="text-base font-semibold mb-2 text-white">{lim.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{lim.desc}</p>
            </div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-gray-900 border border-gray-800 border-l-4 border-l-emerald-500 p-8"
        >
          <h3 className="text-lg font-semibold mb-4 text-white">Future Roadmap</h3>
          <ul className="grid md:grid-cols-2 gap-4 text-sm text-gray-400">
            {futures.map((item, i) => (
              <li key={i} className="flex items-start">
                <span className="text-emerald-500 mr-2">-</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>
    </section>
  );
}
