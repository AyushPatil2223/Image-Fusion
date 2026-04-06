import { useState, useCallback, useRef } from 'react'

const METHOD_LABELS = {
  average:           { name: 'Average Fusion',    icon: '⚖️', color: 'from-blue-500 to-blue-700' },
  max:               { name: 'Max Fusion',         icon: '⬆️', color: 'from-emerald-500 to-emerald-700' },
  gradient_weighted: { name: 'Gradient-Weighted',  icon: '∇',  color: 'from-amber-500 to-amber-700' },
  laplacian_pyramid: { name: 'Laplacian Pyramid',  icon: '🔺', color: 'from-purple-500 to-purple-700' },
  multi_focus_clear: { name: 'Blur+Clear → Clean', icon: '✨', color: 'from-teal-500 to-teal-700' },
  ir_vis_clean: { name: 'IR+Visible → Clean', icon: '🌙', color: 'from-amber-500 to-amber-700' },
  deep_learning:     { name: 'Deep Fusion',        icon: '🧠', color: 'from-indigo-500 to-indigo-700' },
  ir_vis_color:      { name: 'IR+VIS Color',       icon: '🌈', color: 'from-pink-500 to-pink-700' },
  emma:              { name: 'EMMA (CVPR 2024)',   icon: '🔬', color: 'from-cyan-500 to-cyan-700' },
}

export default function ComparePage() {
  const [images,   setImages]  = useState([])
  const [loading,  setLoading] = useState(false)
  const [results,  setResults] = useState(null)
  const [error,    setError]   = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const addFiles = useCallback((files) => {
    const valid = Array.from(files).filter(f => f.type.startsWith('image/'))
    setImages(prev => [...prev, ...valid.map(f => ({
      file: f, url: URL.createObjectURL(f), name: f.name
    }))].slice(0, 6))
    setResults(null)
    setError(null)
  }, [])

  const removeImage = (i) => {
    setImages(prev => { URL.revokeObjectURL(prev[i].url); return prev.filter((_, idx) => idx !== i) })
    setResults(null)
  }

  const handleCompare = async () => {
    if (images.length < 2) { setError('Need at least 2 images.'); return }
    setLoading(true); setError(null); setResults(null)
    try {
      const fd = new FormData()
      images.forEach((img, i) => fd.append(`image${i}`, img.file))
      const res  = await fetch('/api/compare', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok || data.error) throw new Error(data.error || 'Compare failed')
      setResults(data.results)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const best = results
    ? Object.entries(results).reduce((a, [k, v]) =>
        v.metrics.ssim_avg > (results[a]?.metrics.ssim_avg ?? -Infinity) ? k : a,
        Object.keys(results)[0])
    : null

  return (
    <div className="w-full flex flex-col gap-10">
      {/* HERO */}
      <div className="text-center pt-8 pb-4">
        <div className="inline-block px-4 py-1.5 mb-6 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 font-medium text-sm">
          📊 Side-by-Side Comparison
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400">
          Compare All Algorithms
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Upload your images once and see all four fusion algorithms run simultaneously.
          Quality metrics help you pick the best result.
        </p>
      </div>

      {/* Upload Card */}
      <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
        <div className="flex items-center gap-3 mb-6 text-lg font-bold">
          <span className="p-2 rounded bg-white/10">📁</span>
          Source Images
          <span className="ml-auto text-sm bg-white/10 px-3 py-1 rounded-full text-slate-300">
            {images.length}/6
          </span>
        </div>

        <div
          id="compare-upload-zone"
          className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer
            ${dragging
              ? 'border-indigo-400 bg-indigo-500/20 scale-[1.02]'
              : 'border-white/20 bg-white/5 hover:border-indigo-400 hover:bg-indigo-500/10'}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}
        >
          <div className="text-4xl mb-3">🖼️</div>
          <h3 className="text-xl font-bold mb-2">Drop images here or click to browse</h3>
          <p className="text-slate-400 text-sm">Upload 2–6 source images · PNG, JPG, TIFF</p>
          <input ref={inputRef} type="file" accept="image/*" multiple className="hidden"
            onChange={e => addFiles(e.target.files)} />
        </div>

        {images.length > 0 && (
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3 mt-6">
            {images.map((img, i) => (
              <div key={i} className="relative aspect-square rounded-lg overflow-hidden border border-white/10 group bg-black/50">
                <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                <button
                  className="absolute top-1 right-1 w-5 h-5 rounded-full bg-red-500/90 text-white text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={() => removeImage(i)}
                >×</button>
                <span className="absolute bottom-1 left-1 px-2 py-0.5 rounded bg-black/60 text-xs font-medium backdrop-blur-sm">
                  Src {i + 1}
                </span>
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-200">
            ⚠️ {error}
          </div>
        )}

        <div className="flex gap-4 mt-6">
          <button
            id="btn-compare"
            className="flex-1 px-6 py-4 rounded-lg font-semibold text-white bg-indigo-600 hover:bg-indigo-500 transition-all duration-200 shadow-lg shadow-indigo-500/30 disabled:opacity-50 disabled:cursor-not-allowed text-lg"
            disabled={loading || images.length < 2}
            onClick={handleCompare}
          >
            {loading ? '⏳ Running all methods…' : '📊 Compare All Methods'}
          </button>
          {images.length > 0 && (
            <button
              className="px-6 py-4 rounded-lg font-medium text-slate-300 bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
              onClick={() => { setImages([]); setResults(null) }}
            >
              🗑️ Clear
            </button>
          )}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center gap-4 py-16 text-slate-400">
          <div className="w-14 h-14 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
          <p className="text-lg font-medium">Running all 4 fusion algorithms…</p>
          <p className="text-sm text-slate-500">This compares Average, Max, Gradient-Weighted & Laplacian</p>
        </div>
      )}

      {/* Results */}
      {results && !loading && (
        <>
          <div className="text-2xl font-bold text-white flex items-center gap-3">
            <span className="p-2 rounded-lg bg-white/5">📈</span>
            Fusion Results
          </div>

          {best && (
            <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-200 font-medium flex items-center gap-3">
              <span className="text-2xl">✅</span>
              <span>
                Best SSIM: <strong>{METHOD_LABELS[best]?.name}</strong> — SSIM = {results[best].metrics.ssim_avg}
              </span>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {Object.entries(results).map(([id, r], idx) => (
              <div
                key={id}
                id={`compare-result-${id}`}
                className={`bg-white/5 backdrop-blur-sm rounded-2xl overflow-hidden border transition-all duration-300
                  ${best === id
                    ? 'border-emerald-400 shadow-[0_0_30px_rgba(52,211,153,0.25)]'
                    : 'border-white/10 hover:border-white/20'}`}
              >
                <div className="relative">
                  <img
                    src={`data:image/png;base64,${r.image_b64}`}
                    alt={id}
                    className="w-full aspect-square object-contain bg-black/40"
                  />
                  {best === id && (
                    <span className="absolute top-2 left-2 bg-emerald-500/90 text-white text-xs font-bold px-3 py-1 rounded-full">
                      ★ Best SSIM
                    </span>
                  )}
                  <span className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm text-indigo-300 text-xs font-bold px-3 py-1 rounded-full border border-white/10">
                    {r.time_seconds}s
                  </span>
                </div>
                <div className="p-4">
                  <div className="font-bold text-white mb-3 text-sm">
                    {METHOD_LABELS[id]?.icon} {METHOD_LABELS[id]?.name}
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center bg-white/5 rounded-lg p-2">
                      <div className="text-slate-400 mb-1">EN</div>
                      <div className="font-bold text-blue-400">{r.metrics.entropy}</div>
                    </div>
                    <div className="text-center bg-white/5 rounded-lg p-2">
                      <div className="text-slate-400 mb-1">SSIM</div>
                      <div className="font-bold text-cyan-400">{r.metrics.ssim_avg}</div>
                    </div>
                    <div className="text-center bg-white/5 rounded-lg p-2">
                      <div className="text-slate-400 mb-1">MI</div>
                      <div className="font-bold text-purple-400">{r.metrics.mi_avg}</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Summary Table */}
          <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
            <div className="flex items-center gap-3 mb-6 text-lg font-bold">
              <span className="p-2 rounded bg-white/10">📋</span>
              Metrics Summary
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-slate-400 font-semibold">Algorithm</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Entropy (EN)</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Avg SSIM</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Avg MI</th>
                    <th className="text-center py-3 px-4 text-slate-400 font-semibold">Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results).map(([id, r]) => (
                    <tr
                      key={id}
                      className={`border-b border-white/5 transition-colors
                        ${best === id ? 'bg-emerald-500/10 text-emerald-300' : 'text-slate-300 hover:bg-white/5'}`}
                    >
                      <td className="py-3 px-4 font-medium">
                        {METHOD_LABELS[id]?.icon} {METHOD_LABELS[id]?.name} {best === id ? '★' : ''}
                      </td>
                      <td className="py-3 px-4 text-center font-mono">{r.metrics.entropy}</td>
                      <td className="py-3 px-4 text-center font-mono">{r.metrics.ssim_avg}</td>
                      <td className="py-3 px-4 text-center font-mono">{r.metrics.mi_avg}</td>
                      <td className="py-3 px-4 text-center font-mono">{r.time_seconds}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
