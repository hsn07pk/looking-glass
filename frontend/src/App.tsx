import { useState, useEffect, useRef, useCallback } from 'react'
import { Search, Shield, MessageSquare, Radio, Trash2, Send, Camera, Eye, HelpCircle, Settings, X } from 'lucide-react'
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
interface TrackDetection {
  track_id: number; timestamp: number
  x1: number; y1: number; x2: number; y2: number
  class_name: string; score: number
}

function App() {
  const [cameras, setCameras] = useState<CameraData[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResultItem[]>([])
  const [selectedResultIdx, setSelectedResultIdx] = useState(-1)
  const [searching, setSearching] = useState(false)
  const [selectedCam, setSelectedCam] = useState<string | null>(null)
  const [alertRules, setAlertRules] = useState<AlertRule[]>([])
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [alertQuery, setAlertQuery] = useState('')
  const [chatQuery, setChatQuery] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [clock, setClock] = useState(new Date())
  const [showHelp, setShowHelp] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const mainVideoRef = useRef<HTMLVideoElement | null>(null)
  const videoContainerRef = useRef<HTMLDivElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const [videoDims, setVideoDims] = useState({ w: 1920, h: 1080 })
  const [videoRect, setVideoRect] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [trackData, setTrackData] = useState<TrackDetection[]>([])
  const [liveBboxes, setLiveBboxes] = useState<{ x1: number; y1: number; x2: number; y2: number; class_name: string; score: number }[]>([])
  const animRef = useRef<number>(0)

  // Calculate actual video render rect within container (accounting for letterboxing)
  const updateVideoRect = useCallback(() => {
    const vid = mainVideoRef.current
    const container = videoContainerRef.current
    if (!vid || !container) return
    const vw = vid.videoWidth || 1920
    const vh = vid.videoHeight || 1080
    const cw = container.clientWidth
    const ch = container.clientHeight
    const videoAspect = vw / vh
    const containerAspect = cw / ch
    let rw: number, rh: number, rx: number, ry: number
    if (videoAspect > containerAspect) {
      rw = cw; rh = cw / videoAspect; rx = 0; ry = (ch - rh) / 2
    } else {
      rh = ch; rw = ch * videoAspect; ry = 0; rx = (cw - rw) / 2
    }
    setVideoRect({ x: rx, y: ry, w: rw, h: rh })
    setVideoDims({ w: vw, h: vh })
  }, [])

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    fetch(`${API}/cameras`).then(r => r.json()).then(setCameras).catch(() => {})
    fetchAlertRules()
  }, [])

  useEffect(() => {
    try {
      const wsHost = window.location.hostname || 'localhost'
      const ws = new WebSocket(`ws://${wsHost}:8000/alerts/ws`)
      wsRef.current = ws
      ws.onmessage = (e) => {
        const alert = JSON.parse(e.data) as AlertEvent
        setAlerts(prev => [alert, ...prev].slice(0, 50))
      }
      const hb = setInterval(() => { if (ws.readyState === 1) ws.send('ping') }, 30000)
      return () => { clearInterval(hb); ws.close() }
    } catch {}
  }, [])

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMessages])

  // Resize observer for video container
  useEffect(() => {
    const container = videoContainerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => updateVideoRect())
    ro.observe(container)
    return () => ro.disconnect()
  }, [updateVideoRect])

  const fetchAlertRules = async () => {
    try { setAlertRules(await (await fetch(`${API}/alerts/rules`)).json()) } catch {}
  }

  const doSearch = async () => {
    if (!query.trim()) return
    setSearching(true)
    try {
      const data = await (await fetch(`${API}/search`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, top_k: 10 }),
      })).json()
      const newResults = data.results || []
      setResults(newResults)
      setSelectedResultIdx(0)
      if (newResults.length > 0) setSelectedCam(newResults[0].camera_id)
    } catch { setResults([]) }
    setSearching(false)
  }

  const selectResult = (idx: number) => {
    setSelectedResultIdx(idx)
    setSelectedCam(results[idx].camera_id)
    if (mainVideoRef.current && results[idx].timestamp >= 0) {
      mainVideoRef.current.currentTime = results[idx].timestamp
      mainVideoRef.current.play()
    }
  }

  const registerAlert = async () => {
    if (!alertQuery.trim()) return
    await fetch(`${API}/alerts/rules`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: alertQuery }),
    }).catch(() => {})
    setAlertQuery('')
    fetchAlertRules()
  }

  const deleteAlertRule = async (ruleId: string) => {
    await fetch(`${API}/alerts/rules/${ruleId}`, { method: 'DELETE' }).catch(() => {})
    fetchAlertRules()
  }

  const askAnalytics = async () => {
    if (!chatQuery.trim()) return
    const question = chatQuery
    setChatMessages(prev => [...prev, { role: 'user', content: question }])
    setChatQuery('')
    setChatLoading(true)
    try {
      const data = await (await fetch(`${API}/analytics/ask`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: question }),
      })).json()
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.answer || 'No answer.' }])
    } catch {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error connecting to backend.' }])
    }
    setChatLoading(false)
  }

  const activeResult = results[selectedResultIdx] || null

  useEffect(() => {
    if (activeResult && mainVideoRef.current && selectedCam === activeResult.camera_id) {
      mainVideoRef.current.currentTime = activeResult.timestamp
    }
  }, [selectedResultIdx])

  // Fetch tracking data
  useEffect(() => {
    if (selectedCam) {
      fetch(`${API}/cameras/${selectedCam}/tracks`).then(r => r.json()).then(setTrackData).catch(() => setTrackData([]))
    } else { setTrackData([]) }
  }, [selectedCam])

  // Animate bounding boxes with interpolation
  useEffect(() => {
    if (!trackData.length) { setLiveBboxes([]); return }
    const tracks = new Map<number, TrackDetection[]>()
    for (const d of trackData) {
      if (!tracks.has(d.track_id)) tracks.set(d.track_id, [])
      tracks.get(d.track_id)!.push(d)
    }

    const animate = () => {
      const vid = mainVideoRef.current
      if (!vid) { animRef.current = requestAnimationFrame(animate); return }
      const t = vid.currentTime
      const boxes: typeof liveBboxes = []
      for (const [, dets] of tracks) {
        let before: TrackDetection | null = null
        let after: TrackDetection | null = null
        for (const d of dets) {
          if (d.timestamp <= t) before = d
          if (d.timestamp >= t && !after) after = d
        }
        if (!before && !after) continue
        if (before && !after) after = before
        if (!before && after) before = after
        const dt = after!.timestamp - before!.timestamp
        const frac = dt > 0 ? Math.max(0, Math.min(1, (t - before!.timestamp) / dt)) : 0
        const lerp = (a: number, b: number) => a + (b - a) * frac
        if (t < before!.timestamp - 0.5 || t > after!.timestamp + 0.5) continue
        boxes.push({
          x1: lerp(before!.x1, after!.x1), y1: lerp(before!.y1, after!.y1),
          x2: lerp(before!.x2, after!.x2), y2: lerp(before!.y2, after!.y2),
          class_name: before!.class_name, score: before!.score,
        })
      }
      setLiveBboxes(boxes)
      animRef.current = requestAnimationFrame(animate)
    }
    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [trackData])

  // Convert video-pixel bbox to container-pixel position
  const bboxStyle = (box: typeof liveBboxes[0]) => {
    const { w: vw, h: vh } = videoDims
    const { x: rx, y: ry, w: rw, h: rh } = videoRect
    if (rw === 0 || rh === 0) return { display: 'none' as const }
    const scaleX = rw / vw
    const scaleY = rh / vh
    return {
      left: rx + box.x1 * scaleX,
      top: ry + box.y1 * scaleY,
      width: (box.x2 - box.x1) * scaleX,
      height: (box.y2 - box.y1) * scaleY,
    }
  }

  const captionPreview = activeResult?.caption
    ? activeResult.caption.length > 180 ? activeResult.caption.slice(0, 180) + '...' : activeResult.caption
    : null

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] text-[#e5e5e5]">

      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-between px-5 py-2 border-b border-[#1a1a1a] bg-[#0c0c0c]">
        <div className="flex items-center gap-3">
          <Eye size={18} className="text-[#00ff88]" />
          <span className="text-sm font-semibold tracking-[0.15em] uppercase">Looking Glass</span>
          <span className="text-[9px] text-[#444] font-mono tracking-wider">SPRINGINEERING 2026</span>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setShowHelp(true)} className="text-[#444] hover:text-[#888] transition-colors" title="Help"><HelpCircle size={15} /></button>
          <button onClick={() => setShowSettings(true)} className="text-[#444] hover:text-[#888] transition-colors" title="Settings"><Settings size={15} /></button>
          <div className="h-3 w-px bg-[#222]" />
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] live-dot" />
            <span className="text-[#00ff88] font-mono text-[11px] font-medium">{cameras.length} LIVE</span>
          </div>
          <span className="font-mono text-[11px] text-[#555]">{clock.toLocaleTimeString()}</span>
        </div>
      </header>

      {/* Search */}
      <div className="flex-shrink-0 px-5 py-2 border-b border-[#1a1a1a] bg-[#0c0c0c]">
        <div className="flex gap-2 max-w-3xl">
          <div className="flex-1 flex items-center gap-2 bg-[#111] border border-[#222] rounded-lg px-3 py-1.5 focus-within:border-[#00ff88]/40 transition-all">
            <Search size={14} className="text-[#444]" />
            <input
              className="flex-1 bg-transparent outline-none text-sm text-[#e5e5e5] placeholder-[#3a3a3a]"
              placeholder='Search: "orange truck", "person in red jacket", "dog"'
              value={query} onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && doSearch()}
            />
            {query && <button onClick={() => setQuery('')} className="text-[#444] hover:text-[#888] text-[10px]">Clear</button>}
          </div>
          <button onClick={doSearch} disabled={searching}
            className="px-4 py-1.5 bg-[#00ff88] text-[#0a0a0a] text-sm font-semibold rounded-lg hover:bg-[#00ee7d] transition-all disabled:opacity-40">
            {searching ? '...' : 'Search'}
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex overflow-hidden min-h-0">

        {/* Left */}
        <div className="flex-1 flex flex-col min-h-0 overflow-y-auto">

          {/* Camera Grid */}
          <div className="flex-shrink-0 grid grid-cols-4 gap-1 p-2">
            {cameras.map(cam => (
              <div key={cam.camera_id}
                onClick={() => { setSelectedCam(cam.camera_id); setSelectedResultIdx(-1) }}
                className={`cam-tile relative cursor-pointer rounded overflow-hidden border ${
                  selectedCam === cam.camera_id ? 'border-[#00ff88]/50 glow-green' : 'border-[#1a1a1a] hover:border-[#333]'
                }`}>
                <video src={`/videos/${cam.clip_name}`} autoPlay loop muted playsInline className="w-full aspect-video object-cover" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none" />
                <div className="absolute bottom-0 left-0 right-0 px-1.5 py-0.5 flex items-center justify-between">
                  <span className="font-mono text-[9px] text-[#00ff88] font-medium">{cam.camera_id.toUpperCase()}</span>
                  <div className="flex items-center gap-0.5">
                    <div className="w-1 h-1 rounded-full bg-red-500 rec-indicator" />
                    <span className="font-mono text-[7px] text-red-400/80">REC</span>
                  </div>
                </div>
                {results.some(r => r.camera_id === cam.camera_id) && (
                  <div className="absolute top-1 right-1 bg-[#00ff88]/20 rounded px-1 py-px">
                    <span className="font-mono text-[7px] text-[#00ff88] font-medium">MATCH</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Video Viewer */}
          <div ref={videoContainerRef}
            className="flex-1 relative min-h-[300px] mx-2 mb-1 rounded border border-[#1a1a1a] bg-[#080808] overflow-hidden">
            {selectedCam ? (
              <>
                <video
                  ref={mainVideoRef}
                  key={selectedCam}
                  src={`/videos/${cameras.find(c => c.camera_id === selectedCam)?.clip_name}`}
                  autoPlay loop muted playsInline
                  className="absolute inset-0 w-full h-full object-contain"
                  onLoadedMetadata={() => {
                    updateVideoRect()
                    if (activeResult && mainVideoRef.current && selectedCam === activeResult.camera_id) {
                      mainVideoRef.current.currentTime = activeResult.timestamp
                    }
                  }}
                />

                {/* Bounding boxes — positioned relative to actual video render area */}
                {liveBboxes.map((box, i) => {
                  const style = bboxStyle(box)
                  if ('display' in style) return null
                  return (
                    <div key={i} className="absolute bbox-corners pointer-events-none"
                      style={{ ...style, transition: 'left 0.08s linear, top 0.08s linear, width 0.08s linear, height 0.08s linear',
                        boxShadow: '0 0 6px rgba(0,255,136,0.25)' }}>
                      <span className="bbox-corner-tr" />
                      <span className="bbox-corner-bl" />
                      <span className="absolute -top-4 left-0 font-mono text-[9px] bg-black/80 backdrop-blur-sm text-[#00ff88] px-1 py-px rounded-sm whitespace-nowrap border border-[#00ff88]/30">
                        {box.class_name} {(box.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}

                {/* Camera label */}
                <div className="absolute top-2 right-2 flex items-center gap-1 bg-black/60 backdrop-blur-sm rounded px-1.5 py-0.5 border border-[#222] z-10">
                  <Camera size={9} className="text-[#00ff88]" />
                  <span className="font-mono text-[9px] text-[#00ff88]">{selectedCam.toUpperCase()}</span>
                </div>

                {/* Caption */}
                {captionPreview && activeResult && selectedCam === activeResult.camera_id && (
                  <div className="absolute top-2 left-2 max-w-sm bg-black/60 backdrop-blur-sm rounded px-2 py-1 border border-[#222] z-10">
                    <p className="text-[10px] text-white/70 leading-relaxed">{captionPreview}</p>
                  </div>
                )}

                {/* Match info */}
                {activeResult && selectedCam === activeResult.camera_id && (
                  <div className="absolute bottom-2 left-2 flex items-center gap-2 bg-black/70 backdrop-blur-sm rounded px-2 py-1 border border-[#222] z-10">
                    <span className="font-mono text-[11px] font-semibold text-[#00ff88]">{(activeResult.score * 100).toFixed(1)}%</span>
                    <span className="text-[#333]">|</span>
                    <span className="font-mono text-[10px] text-[#666]">{activeResult.timestamp.toFixed(1)}s</span>
                    {activeResult.detections.length > 0 && (
                      <span className="font-mono text-[10px] text-[#555]">{activeResult.detections.length} det</span>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-[#2a2a2a] gap-2">
                <Eye size={28} />
                <span className="text-xs">Select a camera or search to begin</span>
              </div>
            )}
          </div>

          {/* Results strip */}
          {results.length > 0 && (
            <div className="flex-shrink-0 px-2 pb-2">
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-[9px] text-[#444] uppercase">{results.length} results for</span>
                <span className="font-mono text-[9px] text-[#00ff88]">"{query}"</span>
              </div>
              <div className="flex gap-1 overflow-x-auto pb-1">
                {results.map((r, i) => (
                  <button key={i} onClick={() => selectResult(i)}
                    className={`result-card flex-shrink-0 rounded border px-2 py-1 text-left ${
                      selectedResultIdx === i ? 'active' : 'border-[#1a1a1a]'
                    }`}>
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] font-semibold text-[#aaa]">{r.camera_id.toUpperCase()}</span>
                      <span className={`font-mono text-[9px] ${selectedResultIdx === i ? 'text-[#00ff88]' : 'text-[#444]'}`}>
                        {(r.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="font-mono text-[8px] text-[#333] mt-0.5">
                      {r.timestamp.toFixed(1)}s{r.detections.length > 0 ? ` / ${r.detections.length} det` : ''}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar */}
        <div className="w-72 flex-shrink-0 border-l border-[#1a1a1a] flex flex-col bg-[#0b0b0b] overflow-hidden">

          {/* Alerts */}
          <div className="flex-1 flex flex-col border-b border-[#1a1a1a] min-h-0">
            <div className="flex-shrink-0 px-3 py-1.5 flex items-center gap-2 border-b border-[#1a1a1a]">
              <Shield size={11} className="text-[#00ff88]" />
              <span className="text-[10px] font-semibold tracking-wider uppercase">Alerts</span>
              {alerts.length > 0 && (
                <span className="ml-auto font-mono text-[8px] bg-[#00ff88]/15 text-[#00ff88] px-1.5 py-0.5 rounded-full">{alerts.length}</span>
              )}
            </div>
            <div className="flex-shrink-0 px-3 py-1.5">
              <div className="flex gap-1">
                <input className="flex-1 bg-[#111] border border-[#1e1e1e] rounded px-2 py-1 text-[10px] outline-none focus:border-[#00ff88]/30 text-[#ccc] placeholder-[#333]"
                  placeholder="Alert when: person with bag..."
                  value={alertQuery} onChange={e => setAlertQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && registerAlert()} />
                <button onClick={registerAlert} className="px-1.5 py-1 bg-[#00ff88] text-[#0a0a0a] rounded text-[10px] font-bold">+</button>
              </div>
            </div>
            {alertRules.length > 0 && (
              <div className="flex-shrink-0 px-3 pb-1">
                {alertRules.map(rule => (
                  <div key={rule.id} className="flex items-center justify-between py-0.5 group">
                    <span className="text-[9px] text-[#555] truncate font-mono">{rule.query}</span>
                    <button onClick={() => deleteAlertRule(rule.id)} className="text-[#333] hover:text-red-400 ml-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Trash2 size={8} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div className="flex-1 overflow-y-auto px-3 min-h-0">
              {alerts.length === 0 && alertRules.length === 0 && (
                <p className="text-[9px] text-[#2a2a2a] mt-2 text-center">Set alerts to monitor cameras</p>
              )}
              {alerts.map((a, i) => (
                <div key={i} className="alert-entry flex items-start gap-1.5 py-1 border-b border-[#131313]">
                  <Radio size={9} className="text-[#00ff88] mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <span className="font-mono text-[9px] text-[#00ff88]">{a.camera_id.toUpperCase()}</span>
                    <span className="font-mono text-[8px] text-[#333] ml-1">{(a.score * 100).toFixed(0)}%</span>
                    <span className="text-[9px] text-[#444] block truncate">{a.query}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="flex-1 flex flex-col min-h-0">
            <div className="flex-shrink-0 px-3 py-1.5 flex items-center gap-2 border-b border-[#1a1a1a]">
              <MessageSquare size={11} className="text-[#00ff88]" />
              <span className="text-[10px] font-semibold tracking-wider uppercase">Analytics</span>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-2 min-h-0">
              {chatMessages.length === 0 && (
                <div className="mt-1 flex flex-col gap-1">
                  <p className="text-[8px] text-[#2a2a2a] tracking-wider uppercase mb-1">Try asking</p>
                  {['How many people in the lobby?', 'What color clothes on cam03?', 'Count vehicles on cam01'].map(q => (
                    <button key={q} onClick={() => setChatQuery(q)}
                      className="text-[10px] text-left bg-[#0e0e0e] border border-[#1a1a1a] rounded px-2 py-1 hover:border-[#2a2a2a] hover:bg-[#111] transition-all text-[#555]">
                      {q}
                    </button>
                  ))}
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`chat-msg mb-1.5 ${msg.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-lg px-2 py-1 text-[10px] leading-relaxed max-w-[95%] ${
                    msg.role === 'user'
                      ? 'bg-[#00ff88]/8 text-[#00ff88] border border-[#00ff88]/15 rounded-br-sm'
                      : 'bg-[#111] text-[#bbb] border border-[#1a1a1a] rounded-bl-sm'
                  }`}>{msg.content}</div>
                </div>
              ))}
              {chatLoading && (
                <div className="chat-msg mb-1.5">
                  <div className="inline-block rounded-lg px-2 py-1 text-[10px] bg-[#111] border border-[#1a1a1a] text-[#444]">
                    <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <div className="flex-shrink-0 px-3 py-1.5 border-t border-[#1a1a1a]">
              <div className="flex gap-1">
                <input className="flex-1 bg-[#111] border border-[#1e1e1e] rounded px-2 py-1 text-[10px] outline-none focus:border-[#00ff88]/30 text-[#ccc] placeholder-[#333]"
                  placeholder="Ask about the footage..."
                  value={chatQuery} onChange={e => setChatQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && askAnalytics()} />
                <button onClick={askAnalytics} disabled={chatLoading}
                  className="px-1.5 py-1 bg-[#00ff88] text-[#0a0a0a] rounded disabled:opacity-30">
                  <Send size={10} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setShowHelp(false)}>
          <div className="bg-[#111] border border-[#222] rounded-xl max-w-lg w-full mx-4 p-6" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold">Getting Started</h2>
              <button onClick={() => setShowHelp(false)} className="text-[#555] hover:text-white"><X size={16} /></button>
            </div>
            <div className="space-y-4 text-sm text-[#999]">
              <div>
                <h3 className="text-[#00ff88] font-medium text-xs uppercase tracking-wider mb-1">Search</h3>
                <p>Type natural language queries like "find the orange truck" or "person in red jacket". The system searches across all camera feeds using AI vision models.</p>
              </div>
              <div>
                <h3 className="text-[#00ff88] font-medium text-xs uppercase tracking-wider mb-1">Cameras</h3>
                <p>Click any camera tile to view its feed. When you search, matching cameras show a green MATCH badge. Click search result cards at the bottom to jump between matches.</p>
              </div>
              <div>
                <h3 className="text-[#00ff88] font-medium text-xs uppercase tracking-wider mb-1">Alerts</h3>
                <p>Register alert rules like "person with bag" to get notified when matching objects are detected. Alerts fire automatically when new matches are found.</p>
              </div>
              <div>
                <h3 className="text-[#00ff88] font-medium text-xs uppercase tracking-wider mb-1">Analytics Chat</h3>
                <p>Ask questions about the footage: "how many people in the lobby?", "what color is the car on cam06?". The AI analyzes tracking data and scene descriptions to answer.</p>
              </div>
              <div className="pt-2 border-t border-[#222]">
                <p className="text-[10px] text-[#444]">Looking Glass uses SigLIP for search, YOLO-World + ByteTrack for detection and tracking, Florence-2 for grounding, and MiniCPM-V for detailed scene analysis. All models run locally.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setShowSettings(false)}>
          <div className="bg-[#111] border border-[#222] rounded-xl max-w-md w-full mx-4 p-6" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold">Settings</h2>
              <button onClick={() => setShowSettings(false)} className="text-[#555] hover:text-white"><X size={16} /></button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex items-center justify-between py-2 border-b border-[#1a1a1a]">
                <span className="text-[#999]">Show bounding boxes</span>
                <span className="text-[#00ff88] text-xs font-mono">ON</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-[#1a1a1a]">
                <span className="text-[#999]">Show captions overlay</span>
                <span className="text-[#00ff88] text-xs font-mono">ON</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-[#1a1a1a]">
                <span className="text-[#999]">Search results count</span>
                <span className="text-[#888] text-xs font-mono">10</span>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-[#1a1a1a]">
                <span className="text-[#999]">Alert threshold</span>
                <span className="text-[#888] text-xs font-mono">7%</span>
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-[#999]">Analytics model</span>
                <span className="text-[#888] text-xs font-mono">llama3.2:3b</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
