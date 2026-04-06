const METRIC_INFO = [
  {
    key:   'entropy',
    label: 'Entropy (EN)',
    icon:  '📊',
    colorClass: 'from-blue-500 to-blue-600',
    textColor: 'text-blue-400',
    barColor: 'bg-gradient-to-r from-blue-500 to-blue-400',
    desc:  'Information richness of the fused image. Higher = more detail preserved.',
    max: 8,
  },
  {
    key:   'ssim_avg',
    label: 'Avg SSIM',
    icon:  '🔍',
    colorClass: 'from-cyan-500 to-cyan-600',
    textColor: 'text-cyan-400',
    barColor: 'bg-gradient-to-r from-cyan-500 to-cyan-400',
    desc:  'Structural Similarity Index. Closer to 1 = better structural fidelity.',
    max: 1,
  },
  {
    key:   'mi_avg',
    label: 'Avg MI',
    icon:  '🔗',
    colorClass: 'from-purple-500 to-purple-600',
    textColor: 'text-purple-400',
    barColor: 'bg-gradient-to-r from-purple-500 to-purple-400',
    desc:  'Mutual Information between fused and sources. Higher = more source info retained.',
    max: 4,
  },
]

function Bar({ value, max, barColor }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden mt-3">
      <div
        className={`h-full rounded-full transition-all duration-700 ${barColor}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export default function MetricsPanel({ metrics, numSources }) {
  if (!metrics) return null

  return (
    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl shadow-xl p-6">
      <div className="flex items-center gap-3 mb-6 text-lg font-bold text-white">
        <span className="p-2 rounded bg-white/10">📐</span>
        Quality Metrics
        <span className="ml-auto text-sm bg-white/10 px-3 py-1 rounded-full text-slate-300">
          {numSources} source{numSources > 1 ? 's' : ''}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {METRIC_INFO.map(m => (
          <div key={m.key} className="bg-white/5 border border-white/10 rounded-xl p-4 hover:bg-white/10 transition-colors">
            <div className="text-2xl mb-2">{m.icon}</div>
            <div className={`text-3xl font-bold mb-1 ${m.textColor}`}>
              {metrics[m.key] ?? '—'}
            </div>
            <div className="text-sm font-semibold text-white mb-1">{m.label}</div>
            <div className="text-xs text-slate-500 leading-relaxed">{m.desc}</div>
            <Bar value={metrics[m.key] ?? 0} max={m.max} barColor={m.barColor} />
          </div>
        ))}
      </div>

      {/* Per-source SSIM breakdown */}
      {metrics.ssim_per_source?.length > 1 && (
        <div className="mt-4 bg-white/5 border border-white/10 rounded-xl p-4">
          <div className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span>🎯</span> SSIM per Source Image
          </div>
          <div className="flex flex-wrap gap-2">
            {metrics.ssim_per_source.map((v, i) => (
              <span key={i} className="font-mono text-xs px-3 py-1.5 bg-white/5 border border-white/10 rounded-full text-slate-300">
                Src {i + 1}: <span className="text-cyan-400 font-bold">{v}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Per-source MI breakdown */}
      {metrics.mi_per_source?.length > 1 && (
        <div className="mt-3 bg-white/5 border border-white/10 rounded-xl p-4">
          <div className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span>🔗</span> MI per Source Image
          </div>
          <div className="flex flex-wrap gap-2">
            {metrics.mi_per_source.map((v, i) => (
              <span key={i} className="font-mono text-xs px-3 py-1.5 bg-white/5 border border-white/10 rounded-full text-slate-300">
                Src {i + 1}: <span className="text-purple-400 font-bold">{v}</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
