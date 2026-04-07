import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Shield, MessageSquare, Radio, Trash2, Send, Camera, Eye, HelpCircle, Settings, X, Scan, Zap, Maximize2, Minimize2 } from 'lucide-react'
import './index.css'

const API = '/api'

interface CameraData { camera_id: string; clip_name: string }
interface SearchResultItem {
  camera_id: string; timestamp: number; score: number; frame_path: string
  detections: { bbox: number[] | null; class_name: string; score: number }[]
  caption: string
}
interface AlertRule { id: string; query: string; threshold: number; camera_filter: string | null }
interface AlertEvent { rule_id: string; query: string; camera_id: string; timestamp: number; score: number }
interface ChatMessage { role: 'user' | 'assistant'; content: string }
interface TrackDet {
  track_id: number; timestamp: number
  x1: number; y1: number; x2: number; y2: number
  class_name: string; score: number
}

export default function App() {
  const [cameras, setCameras] = useState<CameraData[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResultItem[]>([])
  const [selIdx, setSelIdx] = useState(-1)
  const [searching, setSearching] = useState(false)
  const [cam, setCam] = useState<string | null>(null)
  const [rules, setRules] = useState<AlertRule[]>([])
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [alertQ, setAlertQ] = useState('')
  const [chatQ, setChatQ] = useState('')
  const [msgs, setMsgs] = useState<ChatMessage[]>([])
  const [chatBusy, setChatBusy] = useState(false)
  const [clock, setClock] = useState(new Date())
  const [modal, setModal] = useState<'help' | 'settings' | null>(null)
  const [config, setConfig] = useState({ show_bboxes: true, show_captions: true, search_top_k: 10, bbox_min_confidence: 0.35, vision_model: 'minicpm-v', llm_model: 'llama3.2:3b' })
  const [videoTime, setVideoTime] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)

  const ws = useRef<WebSocket | null>(null)
  const vidRef = useRef<HTMLVideoElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const chatEnd = useRef<HTMLDivElement | null>(null)
  const [vNat, setVNat] = useState({ w: 1920, h: 1080 })
  const [vRect, setVRect] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [tracks, setTracks] = useState<TrackDet[]>([])
  const [boxes, setBoxes] = useState<{ x1: number; y1: number; x2: number; y2: number; cls: string; conf: number }[]>([])
  const raf = useRef(0)

  const syncRect = useCallback(() => {
    const v = vidRef.current, c = boxRef.current
    if (!v || !c) return
    const vw = v.videoWidth || 1920, vh = v.videoHeight || 1080
    const cw = c.clientWidth, ch = c.clientHeight
    const va = vw / vh, ca = cw / ch
    let rw: number, rh: number, rx: number, ry: number
    if (va > ca) { rw = cw; rh = cw / va; rx = 0; ry = (ch - rh) / 2 }
    else { rh = ch; rw = ch * va; ry = 0; rx = (cw - rw) / 2 }
    setVRect({ x: rx, y: ry, w: rw, h: rh }); setVNat({ w: vw, h: vh })
  }, [])

  useEffect(() => { const t = setInterval(() => setClock(new Date()), 1000); return () => clearInterval(t) }, [])

  useEffect(() => {
    fetch(`${API}/cameras`).then(r => r.json()).then(setCameras).catch(() => {})
    fetch(`${API}/settings`).then(r => r.json()).then(setConfig).catch(() => {})
    loadRules()
  }, [])

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  useEffect(() => {
    const c = boxRef.current; if (!c) return
    const ro = new ResizeObserver(syncRect); ro.observe(c); return () => ro.disconnect()
  }, [syncRect])

  useEffect(() => {
    try {
      const s = new WebSocket(`ws://${window.location.hostname || 'localhost'}:8000/alerts/ws`)
      ws.current = s
      s.onmessage = e => { setAlerts(p => [JSON.parse(e.data) as AlertEvent, ...p].slice(0, 40)) }
      const hb = setInterval(() => { if (s.readyState === 1) s.send('ping') }, 25000)
      return () => { clearInterval(hb); s.close() }
    } catch {}
  }, [])

  useEffect(() => {
    if (cam) fetch(`${API}/cameras/${cam}/tracks`).then(r => r.json()).then(setTracks).catch(() => setTracks([]))
    else setTracks([])
  }, [cam])

  useEffect(() => {
    const r = results[selIdx]
    if (r && vidRef.current && cam === r.camera_id) vidRef.current.currentTime = r.timestamp
  }, [selIdx])

  // Keyboard shortcuts
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === '?' || e.key === 'h') setModal('help')
      if (e.key === ',') setModal('settings')
      if (e.key === 'Escape') { setModal(null); setFullscreen(false) }
      if (e.key === 'f') setFullscreen(p => !p)
      if (e.key === '/') { e.preventDefault(); document.querySelector<HTMLInputElement>('input')?.focus() }
      if (e.key === 'ArrowRight' && results.length) { e.preventDefault(); pickResult(Math.min(selIdx + 1, results.length - 1)) }
      if (e.key === 'ArrowLeft' && results.length) { e.preventDefault(); pickResult(Math.max(selIdx - 1, 0)) }
      const n = parseInt(e.key); if (n >= 1 && n <= cameras.length) { setCam(cameras[n - 1].camera_id); setSelIdx(-1) }
    }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [results, selIdx, cameras])

  // Video time
  useEffect(() => {
    const v = vidRef.current; if (!v) return
    const u = () => setVideoTime(v.currentTime)
    v.addEventListener('timeupdate', u); return () => v.removeEventListener('timeupdate', u)
  })

  // Bbox animation
  useEffect(() => {
    if (!tracks.length) { setBoxes([]); return }
    const g = new Map<number, TrackDet[]>()
    for (const d of tracks) { if (!g.has(d.track_id)) g.set(d.track_id, []); g.get(d.track_id)!.push(d) }
    const tick = () => {
      const v = vidRef.current; if (!v) { raf.current = requestAnimationFrame(tick); return }
      const t = v.currentTime, out: typeof boxes = []
      for (const [, ds] of g) {
        let a: TrackDet | null = null, b: TrackDet | null = null
        for (const d of ds) { if (d.timestamp <= t) a = d; if (d.timestamp >= t && !b) b = d }
        if (!a && !b) continue; if (!b) b = a; if (!a) a = b
        const dt = b!.timestamp - a!.timestamp, f = dt > 0 ? Math.max(0, Math.min(1, (t - a!.timestamp) / dt)) : 0
        if (t < a!.timestamp - 0.5 || t > b!.timestamp + 0.5) continue
        const lp = (x: number, y: number) => x + (y - x) * f
        out.push({ x1: lp(a!.x1, b!.x1), y1: lp(a!.y1, b!.y1), x2: lp(a!.x2, b!.x2), y2: lp(a!.y2, b!.y2), cls: a!.class_name, conf: a!.score })
      }
      setBoxes(out); raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf.current)
  }, [tracks])

  const loadRules = async () => { try { setRules(await (await fetch(`${API}/alerts/rules`)).json()) } catch {} }
  const updateConfig = async (key: string, value: string | number | boolean) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    await fetch(`${API}/settings`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key, value }) }).catch(() => {})
  }

  const doSearch = async () => {
    if (!query.trim()) return; setSearching(true)
    try {
      const d = await (await fetch(`${API}/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: query, top_k: config.search_top_k }) })).json()
      const r = d.results || []; setResults(r); setSelIdx(0); if (r.length) setCam(r[0].camera_id)
    } catch { setResults([]) }
    setSearching(false)
  }

  const pickResult = (i: number) => {
    setSelIdx(i); setCam(results[i].camera_id)
    if (vidRef.current) { vidRef.current.currentTime = results[i].timestamp; vidRef.current.play() }
  }

  const addAlert = async () => {
    if (!alertQ.trim()) return
    await fetch(`${API}/alerts/rules`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: alertQ }) }).catch(() => {})
    setAlertQ(''); loadRules()
  }
  const delRule = async (id: string) => { await fetch(`${API}/alerts/rules/${id}`, { method: 'DELETE' }).catch(() => {}); loadRules() }

  const ask = async () => {
    if (!chatQ.trim()) return; const q = chatQ
    setMsgs(p => [...p, { role: 'user', content: q }]); setChatQ(''); setChatBusy(true)
    try {
      const d = await (await fetch(`${API}/analytics/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q }) })).json()
      setMsgs(p => [...p, { role: 'assistant', content: d.answer || 'No answer.' }])
    } catch { setMsgs(p => [...p, { role: 'assistant', content: 'Connection error.' }]) }
    setChatBusy(false)
  }

  const hit = results[selIdx] || null
  const toStyle = (b: typeof boxes[0]) => {
    const { w: nw, h: nh } = vNat, { x: rx, y: ry, w: rw, h: rh } = vRect
    if (!rw || !rh) return null
    return { left: rx + b.x1 * rw / nw, top: ry + b.y1 * rh / nh, width: (b.x2 - b.x1) * rw / nw, height: (b.y2 - b.y1) * rh / nh }
  }
  const fmt = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${Math.floor(s % 60).toString().padStart(2, '0')}`

  return (
    <div className="h-full flex flex-col bg-[#060608] text-[#d4d4d8] grain">

      {/* ═══ HEADER ═══ */}
      <header className="flex-shrink-0 flex items-center justify-between px-5 h-11 border-b border-[#18181b] bg-[#09090b]">
        <div className="flex items-center gap-3">
          <Scan size={18} className="text-[#00ff88]" />
          <span className="font-mono text-xs font-bold tracking-[0.2em] uppercase text-white">Looking Glass</span>
          <span className="text-[10px] text-[#3f3f46] font-mono hidden sm:inline">SPRINGINEERING 2026</span>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setModal('help')} className="text-[#3f3f46] hover:text-[#a1a1aa] transition" title="Help (H)"><HelpCircle size={15} /></button>
          <button onClick={() => setModal('settings')} className="text-[#3f3f46] hover:text-[#a1a1aa] transition" title="Settings (,)"><Settings size={15} /></button>
          <div className="w-px h-4 bg-[#27272a]" />
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-[#00ff88] anim-pulse" />
            <span className="font-mono text-xs font-semibold text-[#00ff88]">{cameras.length} LIVE</span>
          </div>
          <span className="font-mono text-xs text-[#52525b] tabular-nums">{clock.toLocaleTimeString()}</span>
        </div>
      </header>

      {/* ═══ SEARCH ═══ */}
      <div className="flex-shrink-0 px-5 py-2 border-b border-[#18181b] bg-[#09090b]">
        <div className="flex gap-2">
          <div className="flex-1 max-w-2xl flex items-center gap-2 bg-[#0f0f12] border border-[#27272a] rounded-lg px-4 py-2 focus-within:border-[#00ff88]/40 transition-colors">
            <Search size={16} className="text-[#52525b] flex-shrink-0" />
            <input className="flex-1 bg-transparent text-sm text-[#e4e4e7] placeholder-[#3f3f46] font-mono outline-none"
              placeholder='"orange truck"  "person in red jacket"  "dog walking"'
              value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && doSearch()} />
            {query && <button onClick={() => setQuery('')} className="text-[#52525b] hover:text-[#a1a1aa] text-xs font-mono">ESC</button>}
          </div>
          <button onClick={doSearch} disabled={searching}
            className="px-5 py-2 bg-[#00ff88] text-[#09090b] text-sm font-mono font-bold rounded-lg hover:bg-[#00ee7d] transition disabled:opacity-30 tracking-wide">
            {searching ? '...' : 'SEARCH'}
          </button>
        </div>
      </div>

      {/* ═══ BODY ═══ */}
      <div className="flex-1 flex min-h-0 overflow-hidden">

        {/* LEFT PANEL */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">

          {/* Camera strip */}
          <div className="flex-shrink-0 flex items-center gap-1.5 px-4 py-2 bg-[#09090b] border-b border-[#18181b] overflow-x-auto">
            {cameras.map((c, i) => (
              <div key={c.camera_id} onClick={() => { setCam(c.camera_id); setSelIdx(-1) }}
                className={`cam-tile relative cursor-pointer rounded overflow-hidden border flex-shrink-0 ${
                  cam === c.camera_id ? 'border-[#00ff88]/60 glow-sm' : 'border-[#27272a] hover:border-[#3f3f46]'
                }`} style={{ width: 130 }}>
                <video src={`/videos/${c.clip_name}`} autoPlay loop muted playsInline className="w-full h-[68px] object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent pointer-events-none" />
                <div className="absolute bottom-0 inset-x-0 px-2 py-0.5 flex justify-between items-center">
                  <span className="font-mono text-[10px] font-bold text-[#00ff88]">{c.camera_id.toUpperCase()}</span>
                  <div className="flex items-center gap-0.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-red-500 anim-blink" />
                    <span className="font-mono text-[8px] text-red-400/70">REC</span>
                  </div>
                </div>
                {results.some(r => r.camera_id === c.camera_id) && (
                  <div className="absolute top-1 right-1"><Zap size={10} className="text-[#00ff88] drop-shadow-[0_0_4px_rgba(0,255,136,0.5)]" /></div>
                )}
                {/* Number hint */}
                <div className="absolute top-1 left-1 font-mono text-[8px] text-[#3f3f46]">{i + 1}</div>
              </div>
            ))}
          </div>

          {/* VIDEO VIEWER */}
          <div ref={boxRef} className={`flex-1 relative bg-[#040406] overflow-hidden ${fullscreen ? 'fixed inset-0 z-40' : ''}`}>
            {cam ? (
              <>
                <video ref={vidRef} key={cam}
                  src={`/videos/${cameras.find(c => c.camera_id === cam)?.clip_name}`}
                  autoPlay loop muted playsInline
                  className="absolute inset-0 w-full h-full object-contain"
                  onLoadedMetadata={() => { syncRect(); if (hit && vidRef.current && cam === hit.camera_id) vidRef.current.currentTime = hit.timestamp }}
                />

                {/* Bounding boxes */}
                {config.show_bboxes && boxes.map((b, i) => {
                  const s = toStyle(b); if (!s) return null
                  return (
                    <div key={i} className="absolute bbox-bracket pointer-events-none"
                      style={{ ...s, transition: 'left 80ms linear, top 80ms linear, width 80ms linear, height 80ms linear', boxShadow: '0 0 8px rgba(0,255,136,0.15)' }}>
                      <span className="bbox-bracket-tr" /><span className="bbox-bracket-bl" />
                      <span className="absolute -top-5 left-0 font-mono text-[10px] bg-black/80 backdrop-blur-sm text-[#00ff88] px-1.5 py-0.5 rounded border border-[#00ff88]/25 whitespace-nowrap">
                        {b.cls} {(b.conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}

                {/* HUD: camera + time + fullscreen */}
                <div className="absolute top-3 right-3 flex items-center gap-2 z-10">
                  <div className="flex items-center gap-1.5 bg-black/60 backdrop-blur-sm border border-[#27272a] rounded-md px-2 py-1">
                    <Camera size={11} className="text-[#00ff88]" />
                    <span className="font-mono text-[11px] font-bold text-[#00ff88]">{cam.toUpperCase()}</span>
                    <span className="font-mono text-[11px] text-[#52525b] tabular-nums ml-1">{fmt(videoTime)}</span>
                  </div>
                  <button onClick={() => setFullscreen(f => !f)}
                    className="bg-black/60 backdrop-blur-sm border border-[#27272a] rounded-md p-1 text-[#52525b] hover:text-[#00ff88] transition">
                    {fullscreen ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                  </button>
                </div>

                {/* Caption */}
                {config.show_captions && hit && cam === hit.camera_id && hit.caption && (
                  <div className="absolute top-3 left-3 max-w-md bg-black/60 backdrop-blur-sm border border-[#27272a] rounded-md px-3 py-1.5 z-10">
                    <p className="text-[11px] text-[#a1a1aa] leading-relaxed">{hit.caption.length > 200 ? hit.caption.slice(0, 200) + '...' : hit.caption}</p>
                  </div>
                )}

                {/* Match info */}
                {hit && cam === hit.camera_id && (
                  <div className="absolute bottom-3 left-3 flex items-center gap-3 bg-black/70 backdrop-blur-sm border border-[#27272a] rounded-md px-3 py-1.5 z-10">
                    <span className="font-mono text-sm font-bold text-[#00ff88]">{(hit.score * 100).toFixed(1)}%</span>
                    <span className="w-px h-4 bg-[#27272a]" />
                    <span className="font-mono text-xs text-[#71717a]">{hit.timestamp.toFixed(1)}s</span>
                    {hit.detections.length > 0 && (
                      <span className="font-mono text-xs text-[#52525b]">{hit.detections.length} detections</span>
                    )}
                  </div>
                )}
              </>
            ) : (
              /* Empty state — show camera grid larger */
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                <Eye size={40} className="text-[#18181b]" />
                <div className="text-center">
                  <p className="text-sm text-[#3f3f46] mb-1">Select a camera or search to begin</p>
                  <p className="text-xs text-[#27272a]">Press <kbd className="font-mono bg-[#18181b] px-1.5 py-0.5 rounded text-[#52525b]">/</kbd> to search or <kbd className="font-mono bg-[#18181b] px-1.5 py-0.5 rounded text-[#52525b]">1-8</kbd> to select a camera</p>
                </div>
              </div>
            )}
          </div>

          {/* Results strip */}
          {results.length > 0 && (
            <div className="flex-shrink-0 px-4 py-2 bg-[#09090b] border-t border-[#18181b]">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="font-mono text-[10px] text-[#52525b] uppercase tracking-wider">{results.length} results for</span>
                <span className="font-mono text-xs text-[#00ff88] font-semibold">"{query}"</span>
                <span className="font-mono text-[10px] text-[#3f3f46] ml-auto">Use arrow keys to navigate</span>
              </div>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {results.map((r, i) => (
                  <button key={i} onClick={() => pickResult(i)}
                    className={`result-pill flex-shrink-0 rounded-md border px-3 py-1.5 text-left min-w-[90px] ${selIdx === i ? 'active' : 'border-[#27272a]'}`}>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-bold text-[#d4d4d8]">{r.camera_id.toUpperCase()}</span>
                      <span className={`font-mono text-[11px] font-semibold ${selIdx === i ? 'text-[#00ff88]' : 'text-[#52525b]'}`}>{(r.score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="font-mono text-[10px] text-[#3f3f46] mt-0.5">
                      {r.timestamp.toFixed(1)}s{r.detections.length > 0 ? ` / ${r.detections.length} det` : ''}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ═══ RIGHT SIDEBAR ═══ */}
        <div className="w-80 flex-shrink-0 border-l border-[#18181b] flex flex-col bg-[#09090b] overflow-hidden">

          {/* ALERTS */}
          <div className="flex-1 flex flex-col border-b border-[#18181b] min-h-0 overflow-hidden">
            <div className="flex-shrink-0 px-4 py-2 flex items-center gap-2 border-b border-[#18181b]">
              <Shield size={13} className="text-[#00ff88]" />
              <span className="text-xs font-semibold tracking-wider uppercase">Alerts</span>
              {alerts.length > 0 && <span className="ml-auto font-mono text-[10px] bg-[#00ff88]/10 text-[#00ff88] px-2 py-0.5 rounded-full">{alerts.length}</span>}
            </div>

            <div className="flex-shrink-0 px-4 py-2">
              <div className="flex gap-1.5">
                <input className="flex-1 bg-[#0f0f12] border border-[#27272a] rounded-md px-3 py-1.5 text-xs font-mono text-[#d4d4d8] placeholder-[#3f3f46] focus:border-[#00ff88]/30 transition outline-none"
                  placeholder="Watch for: person with bag..." value={alertQ} onChange={e => setAlertQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && addAlert()} />
                <button onClick={addAlert} className="px-2.5 py-1.5 bg-[#00ff88] text-[#09090b] rounded-md text-xs font-bold hover:bg-[#00ee7d] transition">+</button>
              </div>
            </div>

            {rules.length > 0 && (
              <div className="flex-shrink-0 px-4 pb-1.5 space-y-0.5">
                <p className="text-[10px] text-[#3f3f46] uppercase tracking-wider font-mono">Active rules</p>
                {rules.map(r => (
                  <div key={r.id} className="flex items-center justify-between py-0.5 group">
                    <span className="text-xs text-[#71717a] truncate font-mono">{r.query}</span>
                    <button onClick={() => delRule(r.id)} className="text-[#27272a] hover:text-red-400 ml-2 opacity-0 group-hover:opacity-100 transition"><Trash2 size={11} /></button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex-1 overflow-y-auto px-4 min-h-0">
              {!alerts.length && !rules.length && (
                <p className="text-xs text-[#27272a] text-center mt-4">Register a watch rule to receive notifications</p>
              )}
              {alerts.map((a, i) => (
                <div key={i} className="anim-fade-up flex items-start gap-2 py-2 border-b border-[#0f0f12]" style={{ animationDelay: `${i * 40}ms` }}>
                  <Radio size={11} className="text-[#00ff88] mt-0.5 flex-shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-semibold text-[#00ff88]">{a.camera_id.toUpperCase()}</span>
                      <span className="font-mono text-[10px] text-[#3f3f46]">{(a.score * 100).toFixed(0)}%</span>
                    </div>
                    <p className="text-xs text-[#52525b] truncate">{a.query}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ANALYTICS CHAT */}
          <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
            <div className="flex-shrink-0 px-4 py-2 flex items-center gap-2 border-b border-[#18181b]">
              <MessageSquare size={13} className="text-[#00ff88]" />
              <span className="text-xs font-semibold tracking-wider uppercase">Analytics</span>
            </div>

            <div className="flex-1 overflow-y-auto px-4 py-3 min-h-0">
              {!msgs.length && (
                <div className="space-y-1.5">
                  <p className="text-[10px] text-[#3f3f46] font-mono uppercase tracking-wider">Suggested questions</p>
                  {['How many people in the lobby?', 'What color clothes on cam03?', 'Count vehicles on cam01', 'Describe what happened on cam08'].map(q => (
                    <button key={q} onClick={() => setChatQ(q)}
                      className="block w-full text-left text-xs text-[#71717a] bg-[#0f0f12] border border-[#18181b] rounded-md px-3 py-2 hover:border-[#27272a] hover:text-[#a1a1aa] transition">
                      {q}
                    </button>
                  ))}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`anim-fade-up mb-2 ${m.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-lg px-3 py-2 text-xs leading-relaxed max-w-[95%] ${
                    m.role === 'user'
                      ? 'bg-[#00ff88]/8 text-[#00ff88] border border-[#00ff88]/15'
                      : 'bg-[#0f0f12] text-[#a1a1aa] border border-[#18181b]'
                  }`}>{m.content}</div>
                </div>
              ))}
              {chatBusy && (
                <div className="anim-fade-in mb-2">
                  <div className="inline-block rounded-lg px-3 py-2 text-xs bg-[#0f0f12] border border-[#18181b] text-[#3f3f46]">
                    <span className="loading-dots font-mono"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={chatEnd} />
            </div>

            <div className="flex-shrink-0 px-4 py-2 border-t border-[#18181b]">
              <div className="flex gap-1.5">
                <input className="flex-1 bg-[#0f0f12] border border-[#27272a] rounded-md px-3 py-1.5 text-xs font-mono text-[#d4d4d8] placeholder-[#3f3f46] focus:border-[#00ff88]/30 transition outline-none"
                  placeholder="Ask about the footage..." value={chatQ} onChange={e => setChatQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && ask()} />
                <button onClick={ask} disabled={chatBusy}
                  className="px-2 py-1.5 bg-[#00ff88] text-[#09090b] rounded-md disabled:opacity-20 hover:bg-[#00ee7d] transition">
                  <Send size={12} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ MODALS ═══ */}
      {modal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-[3px] z-50 flex items-center justify-center anim-fade-in" onClick={() => setModal(null)}>
          <div className="bg-[#0f0f12] border border-[#27272a] rounded-xl max-w-lg w-full mx-4 shadow-2xl anim-fade-up" onClick={e => e.stopPropagation()}>
            {modal === 'help' ? (
              <div className="p-6">
                <div className="flex justify-between items-center mb-5">
                  <h2 className="text-base font-bold">Getting Started</h2>
                  <button onClick={() => setModal(null)} className="text-[#52525b] hover:text-white transition"><X size={16} /></button>
                </div>
                <div className="space-y-4">
                  {[
                    { icon: <Search size={14} />, title: 'Search', desc: 'Type natural language queries to find objects, people, or events across all cameras.',
                      ex: ['"orange truck"', '"person in red jacket"', '"dog"', '"handshake"'] },
                    { icon: <Camera size={14} />, title: 'Cameras', desc: 'Click any thumbnail or press 1-8 to view a feed. Matching cameras show a bolt icon after search.',
                      ex: [] },
                    { icon: <Shield size={14} />, title: 'Alerts', desc: 'Set watchlists to get notified when matching objects appear in any camera feed.',
                      ex: ['"person with bag"', '"red car"'] },
                    { icon: <MessageSquare size={14} />, title: 'Analytics', desc: 'Ask detailed questions. The AI analyzes scene descriptions, tracking data, and clothing colors.',
                      ex: ['"what color clothes?"', '"count vehicles"'] },
                  ].map(s => (
                    <div key={s.title} className="flex gap-3">
                      <div className="mt-0.5 text-[#00ff88] flex-shrink-0">{s.icon}</div>
                      <div>
                        <h3 className="text-xs font-bold text-[#00ff88] mb-0.5">{s.title}</h3>
                        <p className="text-xs text-[#71717a] leading-relaxed">{s.desc}</p>
                        {s.ex.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1.5">
                            {s.ex.map(e => <span key={e} className="font-mono text-[10px] bg-[#18181b] text-[#52525b] px-2 py-0.5 rounded">{e}</span>)}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  <div className="pt-4 mt-4 border-t border-[#18181b]">
                    <h3 className="text-xs font-bold text-[#52525b] mb-2">Keyboard Shortcuts</h3>
                    <div className="grid grid-cols-2 gap-2">
                      {[['/', 'Search'], ['F', 'Fullscreen'], ['H', 'Help'], [',', 'Settings'], ['1-8', 'Camera'], ['\u2190 \u2192', 'Results'], ['Esc', 'Close']].map(([k, d]) => (
                        <div key={k} className="flex items-center gap-2">
                          <kbd className="font-mono text-[10px] bg-[#18181b] text-[#52525b] px-1.5 py-0.5 rounded min-w-[24px] text-center">{k}</kbd>
                          <span className="text-xs text-[#3f3f46]">{d}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <p className="text-[10px] text-[#27272a] mt-3">SigLIP / YOLO-World / ByteTrack / Florence-2 / MiniCPM-V / Ollama. All local, zero cloud.</p>
                </div>
              </div>
            ) : (
              <div className="p-6">
                <div className="flex justify-between items-center mb-5">
                  <h2 className="text-base font-bold">Settings</h2>
                  <button onClick={() => setModal(null)} className="text-[#52525b] hover:text-white transition"><X size={16} /></button>
                </div>
                <div className="space-y-1">
                  <SettingToggle label="Bounding boxes" value={config.show_bboxes} onToggle={() => updateConfig('show_bboxes', !config.show_bboxes)} />
                  <SettingToggle label="Caption overlay" value={config.show_captions} onToggle={() => updateConfig('show_captions', !config.show_captions)} />
                  <div className="flex items-center justify-between py-3 border-b border-[#18181b]">
                    <span className="text-sm text-[#a1a1aa]">Search results</span>
                    <select value={config.search_top_k} onChange={e => updateConfig('search_top_k', Number(e.target.value))}
                      className="bg-[#18181b] border border-[#27272a] text-xs font-mono text-[#a1a1aa] rounded-md px-2 py-1 outline-none">
                      <option value={5}>5</option><option value={10}>10</option><option value={20}>20</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-[#18181b]">
                    <span className="text-sm text-[#a1a1aa]">Min detection confidence</span>
                    <select value={config.bbox_min_confidence} onChange={e => updateConfig('bbox_min_confidence', Number(e.target.value))}
                      className="bg-[#18181b] border border-[#27272a] text-xs font-mono text-[#a1a1aa] rounded-md px-2 py-1 outline-none">
                      <option value={0.25}>25%</option><option value={0.35}>35%</option><option value={0.5}>50%</option><option value={0.7}>70%</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between py-3 border-b border-[#18181b]">
                    <span className="text-sm text-[#a1a1aa]">Vision model</span>
                    <span className="font-mono text-xs text-[#52525b]">{config.vision_model}</span>
                  </div>
                  <div className="flex items-center justify-between py-3">
                    <span className="text-sm text-[#a1a1aa]">LLM model</span>
                    <span className="font-mono text-xs text-[#52525b]">{config.llm_model}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function SettingToggle({ label, value, onToggle }: { label: string; value: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-[#18181b]">
      <span className="text-sm text-[#a1a1aa]">{label}</span>
      <button onClick={onToggle}
        className={`relative w-9 h-5 rounded-full transition-colors ${value ? 'bg-[#00ff88]' : 'bg-[#27272a]'}`}>
        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${value ? 'translate-x-4' : 'translate-x-0.5'}`} />
      </button>
    </div>
  )
}
