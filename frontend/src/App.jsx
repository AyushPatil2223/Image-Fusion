import { useState, useEffect } from 'react'
import FusePage   from './pages/FusePage.jsx'
import SuperResolutionPage from './pages/SuperResolutionPage.jsx'
import ComparePage from './pages/ComparePage.jsx'
import AboutPage  from './pages/AboutPage.jsx'

const TABS = [
  { id: 'fuse',    label: 'Fuse Images', icon: '⚗️' },
  { id: 'sr',      label: 'Super Resolution', icon: '✨' },
  { id: 'compare', label: 'Compare All', icon: '📊' },
  { id: 'about',   label: 'Datasets & Papers', icon: '📚' },
]

export default function App() {
  const [tab,       setTab]    = useState('fuse')
  const [apiOnline, setApiOnline] = useState(null)

  useEffect(() => {
    fetch('/api/health')
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
  }, [])

  return (
    <div className="flex flex-col min-h-screen">
      {/* NAVBAR */}
      <nav className="sticky top-0 z-50 glass-panel border-b border-white/10 px-6 py-4 mb-8">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <a className="flex items-center gap-3 cursor-pointer" onClick={() => setTab('fuse')}>
            <div className="text-2xl p-2 rounded-lg bg-white/5 border border-white/10 shadow-[0_0_15px_rgba(99,102,241,0.3)]">🔬</div>
            <span className="text-xl font-bold text-gradient">Image Fusion Lab</span>
          </a>

          <div className="flex p-1 bg-black/40 rounded-xl border border-white/5 backdrop-blur-md">
            {TABS.map(t => (
              <button
                key={t.id}
                id={`tab-${t.id}`}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-medium transition-all duration-300 ${
                  tab === t.id 
                    ? 'bg-indigo-600 text-white shadow-[0_4px_15px_rgba(99,102,241,0.4)]' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
                onClick={() => setTab(t.id)}
              >
                {t.icon} <span>{t.label}</span>
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-black/40 border border-white/5 text-sm font-medium">
            <span className={`w-2.5 h-2.5 rounded-full ${apiOnline === false ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]' : apiOnline ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]' : 'bg-yellow-400 animate-pulse'}`} />
            <span className={apiOnline === false ? 'text-red-400' : 'text-slate-300'}>
              {apiOnline === null ? 'Connecting…' : apiOnline ? 'API Online' : 'API Offline'}
            </span>
          </div>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-6 pb-12">
        {tab === 'fuse'    && <FusePage />}
        {tab === 'sr'      && <SuperResolutionPage />}
        {tab === 'compare' && <ComparePage />}
        {tab === 'about'   && <AboutPage />}
      </main>

      {/* FOOTER */}
      <footer className="py-6 text-center text-slate-500 text-sm border-t border-white/5 bg-black/20 backdrop-blur-md">
        <p>
          Image Fusion Lab — Infrared × Visible · Medical · Multi-Exposure |{' '}
          Built with{' '}
          <a href="https://tailwindcss.com" target="_blank" rel="noreferrer" className="text-indigo-400 hover:text-indigo-300 transition-colors">
            Tailwind CSS
          </a>
          {' '}and Advanced Deep Learning Image Fusion
        </p>
      </footer>
    </div>
  )
}
