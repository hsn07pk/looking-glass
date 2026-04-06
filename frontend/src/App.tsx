import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Shield, MessageSquare, Radio, Trash2, Send, Camera, Eye, HelpCircle, Settings, X, Scan, Zap } from 'lucide-react'
import './index.css'

const API = '/api'

// ─── Types ───

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

// ─── App ───

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
  const [config, setConfig] = useState({
    show_bboxes: true, show_captions: true, search_top_k: 10,
    alert_threshold: 0.07, bbox_min_confidence: 0.35,
    vision_model: 'minicpm-v', llm_model: 'llama3.2:3b',
  })

  const ws = useRef<WebSocket | null>(null)
  const vidRef = useRef<HTMLVideoElement | null>(null)
  const boxRef = useRef<HTMLDivElement | null>(null)
  const chatEnd = useRef<HTMLDivElement | null>(null)

  const [vNat, setVNat] = useState({ w: 1920, h: 1080 })
  const [vRect, setVRect] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [tracks, setTracks] = useState<TrackDet[]>([])
  const [boxes, setBoxes] = useState<{ x1: number; y1: number; x2: number; y2: number; cls: string; conf: number }[]>([])
  const raf = useRef(0)
  const [videoTime, setVideoTime] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)

  // ─── Video rect calculation (accounts for object-contain letterboxing) ───

  const syncRect = useCallback(() => {
    const v = vidRef.current, c = boxRef.current
    if (!v || !c) return
    const vw = v.videoWidth || 1920, vh = v.videoHeight || 1080
    const cw = c.clientWidth, ch = c.clientHeight
    const va = vw / vh, ca = cw / ch
    let rw: number, rh: number, rx: number, ry: number
    if (va > ca) { rw = cw; rh = cw / va; rx = 0; ry = (ch - rh) / 2 }
    else { rh = ch; rw = ch * va; ry = 0; rx = (cw - rw) / 2 }
    setVRect({ x: rx, y: ry, w: rw, h: rh })
    setVNat({ w: vw, h: vh })
  }, [])

  // ─── Effects ───

  useEffect(() => { const t = setInterval(() => setClock(new Date()), 1000); return () => clearInterval(t) }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === '?' || (e.key === 'h' && !e.ctrlKey)) { setModal('help'); e.preventDefault() }
      if (e.key === ',' || e.key === 's') { setModal('settings'); e.preventDefault() }
      if (e.key === 'Escape') { setModal(null); setFullscreen(false) }
      if (e.key === 'f') { setFullscreen(p => !p); e.preventDefault() }
      if (e.key === '/' || e.key === 'k') { document.querySelector<HTMLInputElement>('input')?.focus(); e.preventDefault() }
      // Arrow keys to navigate results
      if (e.key === 'ArrowRight' && results.length > 0) { pickResult(Math.min(selIdx + 1, results.length - 1)); e.preventDefault() }
      if (e.key === 'ArrowLeft' && results.length > 0) { pickResult(Math.max(selIdx - 1, 0)); e.preventDefault() }
      // Number keys 1-8 to select cameras
      const num = parseInt(e.key); if (num >= 1 && num <= cameras.length) { setCam(cameras[num - 1].camera_id); setSelIdx(-1) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [results, selIdx, cameras])

  // Track video time
  useEffect(() => {
    const v = vidRef.current; if (!v) return
    const update = () => setVideoTime(v.currentTime)
    v.addEventListener('timeupdate', update)
    return () => v.removeEventListener('timeupdate', update)
  })
  useEffect(() => {
    fetch(`${API}/cameras`).then(r => r.json()).then(setCameras).catch(() => {})
    fetch(`${API}/settings`).then(r => r.json()).then(setConfig).catch(() => {})
    loadRules()
  }, [])
  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  useEffect(() => {
    const c = boxRef.current; if (!c) return
    const ro = new ResizeObserver(syncRect); ro.observe(c)
    return () => ro.disconnect()
  }, [syncRect])

  // WebSocket
  useEffect(() => {
    try {
      const s = new WebSocket(`ws://${window.location.hostname || 'localhost'}:8000/alerts/ws`)
      ws.current = s
      s.onmessage = e => { const a = JSON.parse(e.data) as AlertEvent; setAlerts(p => [a, ...p].slice(0, 40)) }
      const hb = setInterval(() => { if (s.readyState === 1) s.send('ping') }, 25000)
      return () => { clearInterval(hb); s.close() }
    } catch {}
  }, [])

  // Fetch tracks when camera changes
  useEffect(() => {
    if (cam) fetch(`${API}/cameras/${cam}/tracks`).then(r => r.json()).then(setTracks).catch(() => setTracks([]))
    else setTracks([])
  }, [cam])

  // Seek on result select
  useEffect(() => {
    const r = results[selIdx]
    if (r && vidRef.current && cam === r.camera_id) vidRef.current.currentTime = r.timestamp
  }, [selIdx])

  // ─── Bbox animation loop ───

  useEffect(() => {
    if (!tracks.length) { setBoxes([]); return }
    const grouped = new Map<number, TrackDet[]>()
    for (const d of tracks) { if (!grouped.has(d.track_id)) grouped.set(d.track_id, []); grouped.get(d.track_id)!.push(d) }

    const tick = () => {
      const v = vidRef.current; if (!v) { raf.current = requestAnimationFrame(tick); return }
      const t = v.currentTime, out: typeof boxes = []
      for (const [, ds] of grouped) {
        let a: TrackDet | null = null, b: TrackDet | null = null
        for (const d of ds) { if (d.timestamp <= t) a = d; if (d.timestamp >= t && !b) b = d }
        if (!a && !b) continue; if (!b) b = a; if (!a) a = b
        const dt = b!.timestamp - a!.timestamp
        const f = dt > 0 ? Math.max(0, Math.min(1, (t - a!.timestamp) / dt)) : 0
        if (t < a!.timestamp - 0.5 || t > b!.timestamp + 0.5) continue
        const lp = (x: number, y: number) => x + (y - x) * f
        out.push({ x1: lp(a!.x1, b!.x1), y1: lp(a!.y1, b!.y1), x2: lp(a!.x2, b!.x2), y2: lp(a!.y2, b!.y2), cls: a!.class_name, conf: a!.score })
      }
      setBoxes(out); raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [tracks])

  // ─── Helpers ───

  const loadRules = async () => { try { setRules(await (await fetch(`${API}/alerts/rules`)).json()) } catch {} }

  const updateConfig = async (key: string, value: string | number | boolean) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    await fetch(`${API}/settings`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key, value }),
    }).catch(() => {})
  }

  const doSearch = async () => {
    if (!query.trim()) return; setSearching(true)
    try {
      const d = await (await fetch(`${API}/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: query, top_k: 10 }) })).json()
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
    const sx = rw / nw, sy = rh / nh
    return { left: rx + b.x1 * sx, top: ry + b.y1 * sy, width: (b.x2 - b.x1) * sx, height: (b.y2 - b.y1) * sy }
  }

  // ─── Render ───

  return (
    <div className="h-full flex flex-col bg-[#060608] text-[#d4d4d8] grain">

      {/* ═══ Top bar ═══ */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 h-10 border-b border-[#1a1a1f] bg-[#08080b]">
        <div className="flex items-center gap-2.5">
          <Scan size={16} className="text-[#00ff88]" />
          <span className="font-mono text-[11px] font-bold tracking-[0.25em] uppercase text-[#e4e4e7]">Looking Glass</span>
          <span className="font-mono text-[8px] text-[#3f3f46] tracking-[0.15em] ml-1">SPRINGINEERING 2026</span>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => setModal('help')} className="text-[#3f3f46] hover:text-[#71717a] transition"><HelpCircle size={13} /></button>
          <button onClick={() => setModal('settings')} className="text-[#3f3f46] hover:text-[#71717a] transition"><Settings size={13} /></button>
          <div className="w-px h-3.5 bg-[#1a1a1f]" />
          <div className="flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] anim-pulse" />
            <span className="font-mono text-[9px] font-semibold text-[#00ff88] tracking-wider">{cameras.length} LIVE</span>
          </div>
          <span className="font-mono text-[9px] text-[#3f3f46] tabular-nums">{clock.toLocaleTimeString()}</span>
        </div>
      </div>

      {/* ═══ Search ═══ */}
      <div className="flex-shrink-0 px-4 py-1.5 border-b border-[#1a1a1f] bg-[#08080b]">
        <div className="flex gap-1.5 max-w-2xl">
          <div className="flex-1 flex items-center gap-2 bg-[#0c0c0f] border border-[#1a1a1f] rounded-md px-3 py-1 focus-within:border-[#00ff88]/30 transition-colors">
            <Search size={13} className="text-[#3f3f46]" />
            <input className="flex-1 bg-transparent text-[12px] text-[#d4d4d8] font-mono" placeholder='"orange truck"  "person in red jacket"  "dog"'
              value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && doSearch()} />
          </div>
          <button onClick={doSearch} disabled={searching}
            className="px-3 py-1 bg-[#00ff88] text-[#060608] text-[11px] font-mono font-bold rounded-md hover:bg-[#00ee7d] transition disabled:opacity-30 tracking-wider">
            {searching ? '...' : 'SEARCH'}
          </button>
        </div>
      </div>

      {/* ═══ Body ═══ */}
      <div className="flex-1 flex min-h-0">

        {/* ── Main panel ── */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">

          {/* Camera filmstrip — compact horizontal */}
          <div className="flex-shrink-0 flex gap-1 px-3 py-1.5 bg-[#08080b] border-b border-[#1a1a1f] overflow-x-auto">
            {cameras.map(c => (
              <div key={c.camera_id} onClick={() => { setCam(c.camera_id); setSelIdx(-1) }}
                className={`cam-tile relative cursor-pointer rounded-[3px] overflow-hidden border flex-shrink-0 w-[120px] ${
                  cam === c.camera_id ? 'border-[#00ff88]/50 glow-sm' : 'border-[#1a1a1f]'
                }`}>
                <video src={`/videos/${c.clip_name}`} autoPlay loop muted playsInline className="w-full h-[56px] object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent pointer-events-none" />
                <div className="absolute bottom-0 inset-x-0 px-1 py-px flex justify-between items-center">
                  <span className="font-mono text-[7px] font-bold text-[#00ff88] tracking-wider">{c.camera_id.toUpperCase()}</span>
                  <div className="flex items-center gap-0.5"><div className="w-[3px] h-[3px] rounded-full bg-red-500 anim-blink" /><span className="font-mono text-[6px] text-red-400/60">REC</span></div>
                </div>
                {results.some(r => r.camera_id === c.camera_id) && (
                  <div className="absolute top-0.5 right-0.5"><Zap size={8} className="text-[#00ff88] drop-shadow-[0_0_3px_rgba(0,255,136,0.6)]" /></div>
                )}
              </div>
            ))}
          </div>

          {/* ── Video viewer (HERO) ── */}
          <div ref={boxRef} className={`flex-1 relative bg-[#040406] scanline overflow-hidden ${fullscreen ? 'fixed inset-0 z-40' : ''}`}>
            {cam ? (
              <>
                <video ref={vidRef} key={cam}
                  src={`/videos/${cameras.find(c => c.camera_id === cam)?.clip_name}`}
                  autoPlay loop muted playsInline
                  className="absolute inset-0 w-full h-full object-contain"
                  onLoadedMetadata={() => { syncRect(); if (hit && vidRef.current && cam === hit.camera_id) vidRef.current.currentTime = hit.timestamp }}
                />

                {/* Bboxes */}
                {config.show_bboxes && boxes.map((b, i) => {
                  const s = toStyle(b); if (!s) return null
                  return (
                    <div key={i} className="absolute bbox-bracket pointer-events-none"
                      style={{ ...s, transition: 'left 80ms linear, top 80ms linear, width 80ms linear, height 80ms linear', boxShadow: '0 0 8px rgba(0,255,136,0.18)' }}>
                      <span className="bbox-bracket-tr" /><span className="bbox-bracket-bl" />
                      <span className="absolute -top-3.5 left-0 font-mono text-[8px] bg-[#060608]/80 backdrop-blur-sm text-[#00ff88] px-1 py-px border border-[#00ff88]/20 rounded-sm whitespace-nowrap">
                        {b.cls} {(b.conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}

                {/* HUD overlays */}
                <div className="absolute top-2 right-2 flex items-center gap-2 z-10">
                  <div className="flex items-center gap-1 bg-[#060608]/70 backdrop-blur-sm border border-[#1a1a1f] rounded px-1.5 py-0.5">
                    <Camera size={8} className="text-[#00ff88]" />
                    <span className="font-mono text-[8px] font-bold text-[#00ff88] tracking-wider">{cam.toUpperCase()}</span>
                  </div>
                  <div className="bg-[#060608]/70 backdrop-blur-sm border border-[#1a1a1f] rounded px-1.5 py-0.5">
                    <span className="font-mono text-[8px] text-[#52525b] tabular-nums">
                      {Math.floor(videoTime / 60).toString().padStart(2, '0')}:{Math.floor(videoTime % 60).toString().padStart(2, '0')}.{Math.floor((videoTime % 1) * 10)}
                    </span>
                  </div>
                  <button onClick={() => setFullscreen(f => !f)} className="bg-[#060608]/70 backdrop-blur-sm border border-[#1a1a1f] rounded px-1 py-0.5 text-[#3f3f46] hover:text-[#00ff88] transition" title="Fullscreen (F)">
                    <Eye size={8} />
                  </button>
                </div>

                {hit && cam === hit.camera_id && (
                  <div className="absolute bottom-2 left-2 flex items-center gap-2 bg-[#060608]/70 backdrop-blur-sm border border-[#1a1a1f] rounded px-2 py-1 z-10">
                    <span className="font-mono text-[11px] font-bold text-[#00ff88]">{(hit.score * 100).toFixed(1)}%</span>
                    <span className="w-px h-3 bg-[#27272a]" />
                    <span className="font-mono text-[9px] text-[#52525b]">{hit.timestamp.toFixed(1)}s</span>
                    {hit.detections.length > 0 && <span className="font-mono text-[9px] text-[#3f3f46]">{hit.detections.length} det</span>}
                  </div>
                )}

                {config.show_captions && hit && cam === hit.camera_id && hit.caption && (
                  <div className="absolute top-2 left-2 max-w-sm bg-[#060608]/70 backdrop-blur-sm border border-[#1a1a1f] rounded px-2 py-1 z-10">
                    <p className="font-mono text-[8px] text-[#a1a1aa] leading-relaxed">{hit.caption.length > 160 ? hit.caption.slice(0, 160) + '...' : hit.caption}</p>
                  </div>
                )}
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <Eye size={32} className="text-[#1a1a1f]" />
                <span className="font-mono text-[10px] text-[#27272a] tracking-wider">SELECT CAMERA OR SEARCH</span>
              </div>
            )}
          </div>

          {/* Results strip */}
          {results.length > 0 && (
            <div className="flex-shrink-0 px-3 py-1 bg-[#08080b] border-t border-[#1a1a1f]">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-mono text-[8px] text-[#3f3f46] tracking-wider uppercase">{results.length} hits</span>
                <span className="font-mono text-[8px] text-[#00ff88]">"{query}"</span>
              </div>
              <div className="flex gap-1 overflow-x-auto pb-0.5">
                {results.map((r, i) => (
                  <button key={i} onClick={() => pickResult(i)}
                    className={`result-pill flex-shrink-0 rounded-[3px] border px-2 py-0.5 ${selIdx === i ? 'active' : 'border-[#1a1a1f]'}`}>
                    <span className="font-mono text-[9px] font-semibold text-[#a1a1aa]">{r.camera_id.toUpperCase()}</span>
                    <span className={`font-mono text-[8px] ml-1.5 ${selIdx === i ? 'text-[#00ff88]' : 'text-[#3f3f46]'}`}>{(r.score * 100).toFixed(0)}%</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Sidebar ── */}
        <div className="w-64 flex-shrink-0 border-l border-[#1a1a1f] flex flex-col bg-[#08080b]">

          {/* Alerts */}
          <div className="flex-1 flex flex-col border-b border-[#1a1a1f] min-h-0">
            <div className="flex-shrink-0 px-3 py-1.5 flex items-center gap-1.5 border-b border-[#1a1a1f]">
              <Shield size={10} className="text-[#00ff88]" />
              <span className="font-mono text-[9px] font-bold tracking-[0.15em] uppercase">Alerts</span>
              {alerts.length > 0 && <span className="ml-auto font-mono text-[7px] bg-[#00ff88]/10 text-[#00ff88] px-1 rounded-full">{alerts.length}</span>}
            </div>
            <div className="flex-shrink-0 px-2 py-1">
              <div className="flex gap-0.5">
                <input className="flex-1 bg-[#0c0c0f] border border-[#1a1a1f] rounded-[3px] px-1.5 py-0.5 text-[9px] font-mono text-[#d4d4d8] focus:border-[#00ff88]/25 transition"
                  placeholder="person with bag..." value={alertQ} onChange={e => setAlertQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && addAlert()} />
                <button onClick={addAlert} className="px-1 bg-[#00ff88] text-[#060608] rounded-[3px] text-[9px] font-bold">+</button>
              </div>
            </div>
            {rules.length > 0 && (
              <div className="flex-shrink-0 px-2 pb-0.5">
                {rules.map(r => (
                  <div key={r.id} className="flex items-center py-px group">
                    <span className="font-mono text-[8px] text-[#52525b] truncate flex-1">{r.query}</span>
                    <button onClick={() => delRule(r.id)} className="text-[#27272a] hover:text-red-400 opacity-0 group-hover:opacity-100 transition"><Trash2 size={7} /></button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex-1 overflow-y-auto px-2 min-h-0">
              {!alerts.length && !rules.length && <p className="font-mono text-[8px] text-[#1a1a1f] text-center mt-3">No alerts configured</p>}
              {alerts.map((a, i) => (
                <div key={i} className="anim-fade-up flex items-start gap-1 py-1 border-b border-[#0f0f12]" style={{ animationDelay: `${i * 30}ms` }}>
                  <Radio size={7} className="text-[#00ff88] mt-0.5" />
                  <div className="min-w-0">
                    <span className="font-mono text-[8px] font-semibold text-[#00ff88]">{a.camera_id.toUpperCase()}</span>
                    <span className="font-mono text-[7px] text-[#27272a] ml-1">{(a.score * 100).toFixed(0)}%</span>
                    <p className="font-mono text-[8px] text-[#3f3f46] truncate">{a.query}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-shrink-0 px-3 py-1.5 flex items-center gap-1.5 border-b border-[#1a1a1f]">
              <MessageSquare size={10} className="text-[#00ff88]" />
              <span className="font-mono text-[9px] font-bold tracking-[0.15em] uppercase">Analytics</span>
            </div>
            <div className="flex-1 overflow-y-auto px-2 py-1.5 min-h-0">
              {!msgs.length && (
                <div className="flex flex-col gap-0.5 mt-0.5">
                  <p className="font-mono text-[7px] text-[#1a1a1f] tracking-wider uppercase mb-0.5">Suggestions</p>
                  {['How many people in the lobby?', 'What color clothes on cam03?', 'Count vehicles on cam01'].map(q => (
                    <button key={q} onClick={() => setChatQ(q)}
                      className="font-mono text-[9px] text-left text-[#52525b] bg-[#0c0c0f] border border-[#1a1a1f] rounded-[3px] px-2 py-1 hover:border-[#27272a] hover:text-[#71717a] transition">
                      {q}
                    </button>
                  ))}
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`anim-fade-up mb-1 ${m.role === 'user' ? 'text-right' : ''}`} style={{ animationDelay: '50ms' }}>
                  <div className={`inline-block rounded-md px-2 py-1 text-[9px] font-mono leading-relaxed max-w-[95%] ${
                    m.role === 'user'
                      ? 'bg-[#00ff88]/5 text-[#00ff88] border border-[#00ff88]/10'
                      : 'bg-[#0c0c0f] text-[#a1a1aa] border border-[#1a1a1f]'
                  }`}>{m.content}</div>
                </div>
              ))}
              {chatBusy && (
                <div className="anim-fade-in mb-1">
                  <div className="inline-block rounded-md px-2 py-1 text-[9px] font-mono bg-[#0c0c0f] border border-[#1a1a1f] text-[#3f3f46]">
                    <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={chatEnd} />
            </div>
            <div className="flex-shrink-0 px-2 py-1 border-t border-[#1a1a1f]">
              <div className="flex gap-0.5">
                <input className="flex-1 bg-[#0c0c0f] border border-[#1a1a1f] rounded-[3px] px-1.5 py-1 text-[9px] font-mono text-[#d4d4d8] focus:border-[#00ff88]/25 transition"
                  placeholder="Ask about footage..." value={chatQ} onChange={e => setChatQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && ask()} />
                <button onClick={ask} disabled={chatBusy} className="px-1 bg-[#00ff88] text-[#060608] rounded-[3px] disabled:opacity-20 transition"><Send size={9} /></button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ Modals ═══ */}
      {modal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-[2px] z-50 flex items-center justify-center anim-fade-in" onClick={() => setModal(null)}>
          <div className="bg-[#0c0c0f] border border-[#1a1a1f] rounded-lg max-w-lg w-full mx-4 anim-fade-up" onClick={e => e.stopPropagation()}>
            {modal === 'help' ? (
              <div className="p-5">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="font-mono text-sm font-bold tracking-wider">GETTING STARTED</h2>
                  <button onClick={() => setModal(null)} className="text-[#3f3f46] hover:text-white transition"><X size={14} /></button>
                </div>
                <div className="space-y-3">
                  {[
                    { icon: <Search size={12} />, title: 'SEARCH', desc: 'Type natural language queries to find objects, people, or events across all cameras.',
                      examples: ['"orange truck"', '"person in red jacket"', '"dog"', '"handshake"'] },
                    { icon: <Camera size={12} />, title: 'CAMERAS', desc: 'Click any thumbnail to view full-size. Matching cameras show a bolt icon after search. Click result cards to jump between matches.',
                      examples: [] },
                    { icon: <Shield size={12} />, title: 'ALERTS', desc: 'Set watchlists to get notified when matching objects appear. Alerts fire instantly when a match is found.',
                      examples: ['"person with bag"', '"red car"', '"someone taking photo"'] },
                    { icon: <MessageSquare size={12} />, title: 'ANALYTICS', desc: 'Ask detailed questions about the footage. The AI analyzes scene descriptions, tracking data, and visual details.',
                      examples: ['"how many people in lobby?"', '"what color clothes on cam03?"', '"describe the vehicles"'] },
                  ].map(s => (
                    <div key={s.title} className="flex gap-3">
                      <div className="mt-0.5 text-[#00ff88]">{s.icon}</div>
                      <div>
                        <h3 className="font-mono text-[9px] font-bold text-[#00ff88] tracking-[0.15em] mb-0.5">{s.title}</h3>
                        <p className="text-[11px] text-[#71717a] leading-relaxed">{s.desc}</p>
                        {s.examples.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {s.examples.map(ex => (
                              <span key={ex} className="font-mono text-[9px] bg-[#0c0c0f] border border-[#1a1a1f] text-[#52525b] px-1.5 py-0.5 rounded">{ex}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  <div className="pt-3 mt-3 border-t border-[#1a1a1f]">
                    <h3 className="font-mono text-[9px] font-bold text-[#3f3f46] tracking-[0.15em] mb-1">QUICK START</h3>
                    <ol className="text-[10px] text-[#52525b] space-y-1 list-decimal list-inside">
                      <li>Type a search query and press Enter</li>
                      <li>Click a result card to view the matching frame</li>
                      <li>Bounding boxes track detected objects in real-time</li>
                      <li>Ask follow-up questions in Analytics Chat</li>
                      <li>Set alerts for ongoing monitoring</li>
                    </ol>
                  </div>
                  <div className="pt-3 mt-3 border-t border-[#1a1a1f]">
                    <h3 className="font-mono text-[9px] font-bold text-[#3f3f46] tracking-[0.15em] mb-1">KEYBOARD SHORTCUTS</h3>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[9px]">
                      {[
                        ['/', 'Focus search'], ['F', 'Fullscreen video'], ['H', 'Help'],
                        ['S', 'Settings'], ['1-8', 'Select camera'], ['Esc', 'Close/exit'],
                        ['\u2190 \u2192', 'Navigate results'],
                      ].map(([k, d]) => (
                        <div key={k} className="flex items-center gap-2">
                          <kbd className="font-mono text-[8px] bg-[#1a1a1f] text-[#52525b] px-1 py-px rounded min-w-[20px] text-center">{k}</kbd>
                          <span className="text-[#3f3f46]">{d}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="pt-3 mt-3 border-t border-[#1a1a1f]">
                    <p className="font-mono text-[8px] text-[#27272a] leading-relaxed">SigLIP search / YOLO-World + ByteTrack detection / Florence-2 grounding / MiniCPM-V scene analysis / Ollama LLM. All models run locally, zero cloud dependency.</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-5">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="font-mono text-sm font-bold tracking-wider">SETTINGS</h2>
                  <button onClick={() => setModal(null)} className="text-[#3f3f46] hover:text-white transition"><X size={14} /></button>
                </div>
                <div className="space-y-0">
                  {/* Toggle: Bounding boxes */}
                  <div className="flex items-center justify-between py-2.5 border-b border-[#0f0f12]">
                    <span className="text-[11px] text-[#71717a]">Bounding boxes</span>
                    <button onClick={() => updateConfig('show_bboxes', !config.show_bboxes)}
                      className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded transition ${config.show_bboxes ? 'bg-[#00ff88]/15 text-[#00ff88]' : 'bg-[#1a1a1f] text-[#3f3f46]'}`}>
                      {config.show_bboxes ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  {/* Toggle: Captions */}
                  <div className="flex items-center justify-between py-2.5 border-b border-[#0f0f12]">
                    <span className="text-[11px] text-[#71717a]">Caption overlay</span>
                    <button onClick={() => updateConfig('show_captions', !config.show_captions)}
                      className={`font-mono text-[10px] font-bold px-2 py-0.5 rounded transition ${config.show_captions ? 'bg-[#00ff88]/15 text-[#00ff88]' : 'bg-[#1a1a1f] text-[#3f3f46]'}`}>
                      {config.show_captions ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  {/* Search count */}
                  <div className="flex items-center justify-between py-2.5 border-b border-[#0f0f12]">
                    <span className="text-[11px] text-[#71717a]">Search results</span>
                    <select value={config.search_top_k} onChange={e => updateConfig('search_top_k', Number(e.target.value))}
                      className="bg-[#0c0c0f] border border-[#1a1a1f] text-[10px] font-mono text-[#71717a] rounded px-1.5 py-0.5 outline-none">
                      <option value={5}>5</option><option value={10}>10</option><option value={20}>20</option>
                    </select>
                  </div>
                  {/* Bbox confidence */}
                  <div className="flex items-center justify-between py-2.5 border-b border-[#0f0f12]">
                    <span className="text-[11px] text-[#71717a]">Min bbox confidence</span>
                    <select value={config.bbox_min_confidence} onChange={e => updateConfig('bbox_min_confidence', Number(e.target.value))}
                      className="bg-[#0c0c0f] border border-[#1a1a1f] text-[10px] font-mono text-[#71717a] rounded px-1.5 py-0.5 outline-none">
                      <option value={0.25}>25%</option><option value={0.35}>35%</option><option value={0.5}>50%</option><option value={0.7}>70%</option>
                    </select>
                  </div>
                  {/* Models (read-only info) */}
                  <div className="flex items-center justify-between py-2.5 border-b border-[#0f0f12]">
                    <span className="text-[11px] text-[#71717a]">Vision model</span>
                    <span className="font-mono text-[10px] text-[#52525b]">{config.vision_model}</span>
                  </div>
                  <div className="flex items-center justify-between py-2.5">
                    <span className="text-[11px] text-[#71717a]">LLM model</span>
                    <span className="font-mono text-[10px] text-[#52525b]">{config.llm_model}</span>
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
