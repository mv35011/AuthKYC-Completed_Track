import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';

const DEMO_VIDEOS = [
  {
    id: 'real_002',
    src: '/videos/real_002.mp4',
    type: 'Real',
    label: 'Real - Subject 002',
    results: {
      prnu: { score: '0.41', passed: true, description: 'Physical camera detected' },
      moire: { score: '6204', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '72', snr: '12.4', passed: true, description: '72 BPM — healthy resting HR' },
      ftca: { score: '0.12', passed: true, description: 'Genuine face confirmed' },
      decision: '✅ APPROVED: Genuine Live Human',
      risk: 0
    }
  },
  {
    id: 'real_012',
    src: '/videos/real_012.mp4',
    type: 'Real',
    label: 'Real - Subject 012',
    results: {
      prnu: { score: '0.29', passed: true, description: 'Weak but valid fingerprint' },
      moire: { score: '5843', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '65', snr: '7.1', passed: true, description: '65 BPM — lower SNR due to lighting' },
      ftca: { score: '0.22', passed: true, description: 'Genuine face confirmed' },
      decision: '✅ APPROVED: Genuine Live Human',
      risk: 0
    }
  },
  {
    id: 'real_013',
    src: '/videos/real_013.mp4',
    type: 'Real',
    label: 'Real - Subject 013',
    results: {
      prnu: { score: '0.52', passed: true, description: 'Strong sensor fingerprint' },
      moire: { score: '7158', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '78', snr: '14.2', passed: true, description: '78 BPM — clear pulse signal' },
      ftca: { score: '0.06', passed: true, description: 'Genuine face confirmed' },
      decision: '✅ APPROVED: Genuine Live Human',
      risk: 0
    }
  },
  {
    id: 'fake_001_870',
    src: '/videos/fake_001_870.mp4',
    type: 'Fake',
    label: 'FaceSwap - 001→870',
    results: {
      prnu: { score: '0.31', passed: true, description: 'Source video retains camera noise' },
      moire: { score: '5102', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '58', snr: '3.4', passed: false, description: 'Weak signal — residual from source' },
      ftca: { score: '0.94', passed: false, description: 'Deepfake artifacts detected' },
      decision: '❌ DENIED: AI-Generated Deepfake',
      risk: 3
    }
  },
  {
    id: 'fake_034_590',
    src: '/videos/fake_034_590.mp4',
    type: 'Fake',
    label: 'Deepfake - 034→590',
    results: {
      prnu: { score: '0.14', passed: false, description: 'Sensor fingerprint destroyed by synthesis' },
      moire: { score: '4836', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '8', snr: '0.8', passed: false, description: 'No biological pulse detected' },
      ftca: { score: '0.97', passed: false, description: 'High-confidence deepfake detection' },
      decision: '❌ DENIED: AI-Generated + No Liveness',
      risk: 5
    }
  },
  {
    id: 'fake_035_036',
    src: '/videos/fake_035_036.mp4',
    type: 'Fake',
    label: 'Face2Face - 035→036',
    results: {
      prnu: { score: '0.22', passed: true, description: 'Partial camera noise preserved' },
      moire: { score: '6147', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '71', snr: '5.9', passed: true, description: 'Source pulse partially transferred' },
      ftca: { score: '0.89', passed: false, description: 'Temporal inconsistencies detected' },
      decision: '⚠️ REVIEW: FTCA Flagged as Deepfake',
      risk: 2
    }
  },
  {
    id: 'fake_036_035',
    src: '/videos/fake_036_035.mp4',
    type: 'Fake',
    label: 'FaceSwap - 036→035',
    results: {
      prnu: { score: '0.09', passed: false, description: 'Virtual/synthetic source detected' },
      moire: { score: '5920', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '12', snr: '1.9', passed: false, description: 'Incoherent pulse signal' },
      ftca: { score: '0.98', passed: false, description: 'Frequency artifacts detected' },
      decision: '❌ DENIED: AI-Generated + Virtual Camera',
      risk: 5
    }
  },
  {
    id: 'fake_427_637',
    src: '/videos/fake_427_637.mp4',
    type: 'Fake',
    label: 'NeuralTextures - 427→637',
    results: {
      prnu: { score: '0.35', passed: true, description: 'Source video camera noise intact' },
      moire: { score: '4512', passed: true, description: 'No replay patterns' },
      rppg: { bpm: '0', snr: '0.3', passed: false, description: 'No biological signal found' },
      ftca: { score: '0.91', passed: false, description: 'Neural texture artifacts detected' },
      decision: '❌ DENIED: AI-Generated Deepfake',
      risk: 3
    }
  }
];

const VideoCard = ({ video, isSelected, onClick }) => {
  const videoRef = useRef(null);

  const handleMouseEnter = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(() => {});
    }
  };

  const handleMouseLeave = () => {
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0.5;
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0.5; // Show a visible frame as thumbnail
    }
  };

  return (
    <div 
      className={`relative cursor-pointer rounded-lg overflow-hidden border transition-all duration-200 ${isSelected ? 'border-indigo-500 ring-2 ring-indigo-500/50 bg-gray-800' : 'border-gray-800 bg-gray-900 hover:border-gray-600'}`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="aspect-[4/3] bg-black relative">
        <video 
          ref={videoRef}
          src={video.src}
          className="w-full h-full object-cover"
          muted
          loop
          playsInline
          preload="metadata"
          onLoadedMetadata={handleLoadedMetadata}
        >
          <source src={video.src} type="video/mp4" />
        </video>
        <div className="absolute top-2 right-2 flex gap-2">
          <span className={`px-2 py-0.5 text-xs font-semibold rounded ${video.type === 'Real' ? 'bg-green-900/80 text-green-300' : 'bg-red-900/80 text-red-300'}`}>
            {video.type}
          </span>
        </div>
      </div>
      <div className="p-3">
        <h4 className="text-sm font-medium text-gray-200">{video.label}</h4>
      </div>
    </div>
  );
};

const ResultsPanel = ({ video }) => {
  if (!video) return null;

  const results = video.results;
  const isReal = video.type === 'Real';

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 lg:p-8 flex flex-col gap-6">
      <div className="flex flex-col md:flex-row gap-6 md:items-start">
        <div className="w-full md:w-1/3 aspect-[4/3] rounded overflow-hidden bg-black border border-gray-800">
          <video 
            src={video.src}
            className="w-full h-full object-cover"
            autoPlay
            muted
            loop
            playsInline
          />
        </div>
        
        <div className="w-full md:w-2/3 flex flex-col justify-center">
          <h3 className="text-xl font-semibold text-white mb-2">{video.label}</h3>
          
          <div className="flex items-center gap-4 mb-6">
            <div className="text-sm text-gray-400">Analysis Result:</div>
            <div className={`px-4 py-2 rounded font-medium ${isReal ? 'bg-green-900/30 text-green-400 border border-green-800' : 'bg-red-900/30 text-red-400 border border-red-800'}`}>
              {results.decision}
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between items-center text-sm">
              <span className="text-gray-400">Risk Score ({results.risk}/7)</span>
              <span className="font-mono text-gray-300">{((results.risk / 7) * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full ${isReal ? 'bg-green-500' : 'bg-red-500'}`} 
                style={{ width: `${(results.risk / 7) * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-gray-300">
          <thead className="bg-gray-800/50 text-gray-400 uppercase text-xs">
            <tr>
              <th className="px-4 py-3 rounded-tl">Module</th>
              <th className="px-4 py-3">Metric</th>
              <th className="px-4 py-3">Score</th>
              <th className="px-4 py-3 rounded-tr">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            <tr className="bg-gray-900/30">
              <td className="px-4 py-3 font-medium">Sensor Noise (PRNU)</td>
              <td className="px-4 py-3 text-gray-500">{results.prnu.description}</td>
              <td className="px-4 py-3 font-mono">{results.prnu.score}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded text-xs ${results.prnu.passed ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {results.prnu.passed ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
            <tr className="bg-gray-800/20">
              <td className="px-4 py-3 font-medium">Recapture (Moiré)</td>
              <td className="px-4 py-3 text-gray-500">{results.moire.description}</td>
              <td className="px-4 py-3 font-mono">{results.moire.score}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded text-xs ${results.moire.passed ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {results.moire.passed ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
            <tr className="bg-gray-900/30">
              <td className="px-4 py-3 font-medium">Liveness (rPPG)</td>
              <td className="px-4 py-3 text-gray-500">{results.rppg.description}</td>
              <td className="px-4 py-3 font-mono">BPM: {results.rppg.bpm}, SNR: {results.rppg.snr}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded text-xs ${results.rppg.passed ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {results.rppg.passed ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
            <tr className="bg-gray-800/20">
              <td className="px-4 py-3 font-medium">Artifacts (FTCA)</td>
              <td className="px-4 py-3 text-gray-500">{results.ftca.description}</td>
              <td className="px-4 py-3 font-mono">{results.ftca.score}</td>
              <td className="px-4 py-3">
                <span className={`px-2 py-1 rounded text-xs ${results.ftca.passed ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                  {results.ftca.passed ? 'PASS' : 'FAIL'}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default function Demo() {
  const [selectedVideo, setSelectedVideo] = useState(DEMO_VIDEOS[0]);

  return (
    <section id="demo" className="min-h-screen bg-[#0a0a1a] py-24 text-gray-200 flex items-center">
      <div className="max-w-7xl mx-auto px-6 w-full">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="w-full"
        >
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-12">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">Demo</h2>
              <p className="text-gray-400 max-w-2xl text-lg">
                Pre-computed analysis results on benchmark videos
              </p>
            </div>
            
            <a 
              href="https://huggingface.co/spaces/mv350113/authkyc-demo" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-indigo-600/10 text-indigo-400 hover:bg-indigo-600/20 border border-indigo-500/20 rounded-lg text-sm font-medium transition-colors"
            >
              Try on HuggingFace Spaces <span className="ml-2">→</span>
            </a>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {DEMO_VIDEOS.map((video) => (
              <VideoCard 
                key={video.id} 
                video={video} 
                isSelected={selectedVideo.id === video.id}
                onClick={() => setSelectedVideo(video)} 
              />
            ))}
          </div>

          <ResultsPanel video={selectedVideo} />
          
          <div className="mt-12 text-center text-sm text-gray-500">
            Videos from FaceForensics++ C23 dataset. Live inference available on HuggingFace Spaces.
          </div>
        </motion.div>
      </div>
    </section>
  );
}
