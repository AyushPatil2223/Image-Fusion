const DATASETS = [
  { name: 'TNO',      modality: 'IR + Visible',   size: '261 sequences',  use: 'Classic benchmark (military scenes)', venue: 'IVIF-ZOO repo' },
  { name: 'MSRS',     modality: 'Multi-Spectral',  size: '500+ MB',        use: 'Road scenes, CDDFuse training',       venue: 'CDDFuse GitHub' },
  { name: 'M3FD',     modality: 'IR + Visible',    size: '33 603 objects', use: 'Object detection + fusion',           venue: 'IVIF-ZOO repo' },
  { name: 'VLF',      modality: 'Vision-Language', size: '8 datasets',     use: 'Language-guided fusion (FILM)',        venue: 'Google Drive'  },
  { name: 'RoadScene',modality: 'IR + Visible',    size: '221 pairs',      use: 'Urban driving scenes',                venue: 'IVIF-ZOO repo' },
  { name: 'FMB',      modality: 'IR + Visible',    size: '1500 pairs',     use: 'Fine-grained multi-band',             venue: 'IVIF-ZOO repo' },
]

const PAPERS = [
  { name: 'CDDFuse', venue: 'CVPR 2023', url: 'https://arxiv.org/abs/2211.14461',
    desc: 'Correlation-Driven Dual-Branch Feature Decomposition for multi-modal image fusion. Transformer-CNN hybrid with INN blocks. State-of-the-art baseline.' },
  { name: 'DDFM (Oral)', venue: 'ICCV 2023', url: 'https://arxiv.org/abs/2303.06840',
    desc: 'Denoising Diffusion Model for multi-modality fusion. Higher quality than CNN methods but requires more compute.' },
  { name: 'EMMA', venue: 'CVPR 2024', url: 'https://arxiv.org/abs/2305.11443',
    desc: 'Equivariant Multi-Modality Image Fusion. Latest SOTA leveraging symmetry constraints.' },
  { name: 'FILM', venue: 'ICML 2024', url: 'https://arxiv.org/abs/2402.02235',
    desc: 'Language-guided fusion via vision-language model (CLIP/ChatGPT prompts). Cross-attention alignment for semantic guidance.' },
  { name: 'DenseFuse', venue: 'IEEE TIP 2019', url: '#',
    desc: 'Highly cited dense encoder-decoder architecture for infrared and visible image fusion. Classic baseline.' },
  { name: 'Survey', venue: 'IEEE TPAMI 2024', url: '#',
    desc: 'Comprehensive survey — From Data Compatibility to Task Adaptation. Covers 100+ methods.' },
]

const ARCH_STEPS = [
  { step: '1', label: 'Encoder',      icon: '🔍', colorFrom: 'from-blue-500',    colorTo: 'to-blue-700',
    desc: 'Restormer-based shared encoder extracts multi-scale features from each source image.' },
  { step: '2', label: 'Decomposition',icon: '⚡', colorFrom: 'from-purple-500',  colorTo: 'to-purple-700',
    desc: 'Lite Transformer (low-freq global) + INN blocks (high-freq local) decompose features per modality.' },
  { step: '3', label: 'Fusion Layer', icon: '⚗️', colorFrom: 'from-cyan-500',    colorTo: 'to-cyan-700',
    desc: 'Correlation-driven loss suppresses redundant cross-modal information before merging.' },
  { step: '4', label: 'Decoder',      icon: '✨', colorFrom: 'from-emerald-500', colorTo: 'to-emerald-700',
    desc: 'CNN decoder reconstructs the final fused image from the merged feature map.' },
]

const PIPELINE_STEPS = [
  { n:'1', title:'Resize & Align',       icon:'📐', desc:'All sources resized to a common resolution (max 1024 px) using LANCZOS resampling.' },
  { n:'2', title:'Activity Maps',         icon:'🗺️', desc:'Per-pixel saliency from LoG, Sobel gradients, and local variance — combined into soft weight maps.' },
  { n:'3', title:'Laplacian Pyramid',     icon:'🔺', desc:'Multi-scale weighted Laplacian pyramid fusion (depth 6) applied channel-by-channel.' },
  { n:'4', title:'YCbCr Colour Blend',   icon:'🎨', desc:'Fused luminance (Y) + saturation-weighted chrominance from colour sources.' },
  { n:'5', title:'Bilateral Denoise',     icon:'🧹', desc:'Edge-preserving bilateral filter removes noise while keeping hard edges crisp.' },
  { n:'6', title:'Retinex Tone Map',      icon:'☀️', desc:'Single-scale Retinex reveals detail in shadows and highlights.' },
  { n:'7', title:'CLAHE (clip=3.0)',      icon:'📈', desc:'Strong local contrast enhancement in LAB space for vivid detail.' },
  { n:'8', title:'Unsharp Mask ×2.0',    icon:'🔬', desc:'Aggressive selective sharpening — only edge pixels are enhanced.' },
  { n:'9', title:'Saturation +30%',       icon:'🌈', desc:'Colour saturation boost via PIL ImageEnhance for vivid, non-washed output.' },
  { n:'10',title:'CNN Refinement',        icon:'🧠', desc:'Optional 4-layer residual CNN (if PyTorch available) for micro-artefact removal.' },
]

export default function AboutPage() {
  return (
    <div className="w-full flex flex-col gap-12 pb-8">
      {/* HERO */}
      <div className="text-center pt-8 pb-4">
        <div className="inline-block px-4 py-1.5 mb-6 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 font-medium text-sm">
          📚 Research Reference
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
          Datasets &amp; Papers
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          All datasets, key research papers, and architecture details used in this project.
        </p>
      </div>

      {/* Fusion Pipeline */}
      <section>
        <div className="flex items-center gap-3 mb-6 text-2xl font-bold text-white">
          <span className="p-2 rounded-lg bg-white/5">⚙️</span>
          Our Fusion Pipeline (10 Steps)
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {PIPELINE_STEPS.map(s => (
            <div key={s.n} className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-xs font-bold bg-indigo-500/30 border border-indigo-400/30 text-indigo-300 rounded-full w-6 h-6 flex items-center justify-center">{s.n}</span>
                <span className="text-lg">{s.icon}</span>
              </div>
              <div className="font-bold text-white text-sm mb-1">{s.title}</div>
              <div className="text-xs text-slate-400 leading-relaxed">{s.desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CDDFuse Architecture */}
      <section>
        <div className="flex items-center gap-3 mb-6 text-2xl font-bold text-white">
          <span className="p-2 rounded-lg bg-white/5">🏗️</span>
          CDDFuse Architecture (CVPR 2023)
        </div>
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
          <p className="text-slate-400 text-sm leading-relaxed mb-6">
            CDDFuse uses a <strong className="text-white">dual-branch Transformer-CNN encoder</strong> — a Restormer-based
            shared feature encoder with a Lite Transformer block for low-frequency global features and Invertible Neural Network (INN) blocks
            for high-frequency local detail, with a correlation-driven decomposition loss.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {ARCH_STEPS.map(s => (
              <div key={s.step} className={`bg-white/5 rounded-xl p-4 border-t-2 bg-gradient-to-b ${s.colorFrom}/10 border ${s.colorFrom}/50`}>
                <div className="text-2xl mb-2">{s.icon}</div>
                <div className={`font-bold text-sm mb-2 text-transparent bg-clip-text bg-gradient-to-r ${s.colorFrom} ${s.colorTo}`}>
                  Step {s.step} — {s.label}
                </div>
                <div className="text-xs text-slate-400 leading-relaxed">{s.desc}</div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-3">
            <a href="https://github.com/Zhaozixiang1228/MMIF-CDDFuse" target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-white bg-indigo-600 hover:bg-indigo-500 transition-colors text-sm">
              🔗 GitHub (CDDFuse)
            </a>
            <a href="https://arxiv.org/abs/2211.14461" target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-slate-300 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm">
              📄 arXiv Paper
            </a>
            <a href="https://github.com/RollingPlain/IVIF_ZOO" target="_blank" rel="noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-slate-300 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors text-sm">
              🗂️ IVIF-ZOO (Datasets)
            </a>
          </div>
        </div>
      </section>

      {/* Datasets */}
      <section>
        <div className="flex items-center gap-3 mb-6 text-2xl font-bold text-white">
          <span className="p-2 rounded-lg bg-white/5">🗂️</span>
          Benchmark Datasets
        </div>
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="text-left py-3 px-4 text-slate-400 font-semibold">Dataset</th>
                <th className="text-left py-3 px-4 text-slate-400 font-semibold">Modality</th>
                <th className="text-left py-3 px-4 text-slate-400 font-semibold">Size</th>
                <th className="text-left py-3 px-4 text-slate-400 font-semibold">Primary Use</th>
                <th className="text-left py-3 px-4 text-slate-400 font-semibold">Source</th>
              </tr>
            </thead>
            <tbody>
              {DATASETS.map(d => (
                <tr key={d.name} className="border-b border-white/5 text-slate-300 hover:bg-white/5 transition-colors">
                  <td className="py-3 px-4 font-bold text-white">{d.name}</td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-xs font-medium">
                      {d.modality}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-mono text-xs text-slate-400">{d.size}</td>
                  <td className="py-3 px-4 text-slate-400 text-xs">{d.use}</td>
                  <td className="py-3 px-4 text-indigo-400 text-xs">{d.venue}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Key Papers */}
      <section>
        <div className="flex items-center gap-3 mb-6 text-2xl font-bold text-white">
          <span className="p-2 rounded-lg bg-white/5">📄</span>
          Key Research Papers
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {PAPERS.map(p => (
            <div key={p.name} className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-5 flex flex-col gap-3 hover:bg-white/10 transition-colors">
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-bold text-white">{p.name}</h3>
                <span className="px-2 py-0.5 rounded-full bg-purple-500/20 border border-purple-400/30 text-purple-300 text-xs font-medium whitespace-nowrap">
                  {p.venue}
                </span>
              </div>
              <p className="text-slate-400 text-sm leading-relaxed flex-1">{p.desc}</p>
              {p.url !== '#' && (
                <a href={p.url} target="_blank" rel="noreferrer"
                  className="inline-flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium text-slate-300 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors w-fit">
                  📄 Read Paper
                </a>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Quick Start */}
      <section>
        <div className="flex items-center gap-3 mb-6 text-2xl font-bold text-white">
          <span className="p-2 rounded-lg bg-white/5">🚀</span>
          Pipeline Quick Start
        </div>
        <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-indigo-400 font-semibold mb-3 text-sm">Project Structure</h3>
              <pre className="bg-black/40 rounded-xl p-4 font-mono text-xs text-slate-300 leading-relaxed border border-white/10 overflow-x-auto">
{`image-fusion-project/
├── backend/
│   ├── app.py          ← Flask API
│   ├── fusion_model.py ← 10-step pipeline
│   ├── requirements.txt
│   ├── uploads/
│   └── results/
└── frontend/
    ├── src/
    │   ├── pages/
    │   │   ├── FusePage.jsx
    │   │   ├── ComparePage.jsx
    │   │   └── AboutPage.jsx
    │   ├── components/
    │   │   └── MetricsPanel.jsx
    │   ├── App.jsx
    │   └── index.css
    └── vite.config.js`}
              </pre>
            </div>
            <div>
              <h3 className="text-indigo-400 font-semibold mb-3 text-sm">Run Locally</h3>
              <pre className="bg-black/40 rounded-xl p-4 font-mono text-xs text-slate-300 leading-relaxed border border-white/10 overflow-x-auto">
{`# 1. Start backend (Flask)
cd backend
pip install -r requirements.txt
python app.py   # → http://localhost:5000

# 2. Start frontend (Vite)
cd frontend
npm install
npm run dev     # → http://localhost:3000

# 3. Open browser
http://localhost:3000`}
              </pre>
              <div className="mt-4 p-3 rounded-xl bg-blue-500/10 border border-blue-500/30 text-blue-200 text-xs flex items-start gap-2">
                <span>💡</span>
                <span>The Vite dev server proxies <code className="bg-black/30 px-1 py-0.5 rounded">/api/*</code> to Flask on port 5000 automatically — no CORS config needed.</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
