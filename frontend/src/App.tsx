import { useState, useEffect, useRef } from 'react'
import { Search, Shield, MessageSquare, Radio, Trash2, Send, Camera, Activity, Eye } from 'lucide-react'
import './index.css'

const API = '/api'

interface CameraData {
  camera_id: string
  clip_name: string
}

interface SearchResultItem {
  camera_id: string
  timestamp: number
  score: number
  frame_path: string
  detections: { bbox: number[] | null; class_name: string; score: number }[]
  caption: string
}

interface AlertRule {
  id: string
  query: string
  threshold: number
  camera_filter: string | null
}

interface AlertEvent {
  rule_id: string
  query: string
  camera_id: string
  timestamp: number
  score: number
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface TrackDetection {
  track_id: number
  timestamp: number
  x1: number; y1: number; x2: number; y2: number
  class_name: string
  score: number
}

function App() {
  const [cameras, setCameras] = useState<CameraData[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResultItem[]>([])
  const [selectedResultIdx, setSelectedResultIdx] = useState<number>(-1)
  const [searching, setSearching] = useState(false)
  const [selectedCam, setSelectedCam] = useState<string | null>(null)
  const [alertRules, setAlertRules] = useState<AlertRule[]>([])
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [alertQuery, setAlertQuery] = useState('')
  const [chatQuery, setChatQuery] = useState('')
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [clock, setClock] = useState(new Date())
  const wsRef = useRef<WebSocket | null>(null)
  const mainVideoRef = useRef<HTMLVideoElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const [videoDims, setVideoDims] = useState<{ w: number; h: number }>({ w: 1920, h: 1080 })
  const [trackData, setTrackData] = useState<TrackDetection[]>([])
  const [liveBboxes, setLiveBboxes] = useState<{ x1: number; y1: number; x2: number; y2: number; class_name: string; score: number }[]>([])
  const animRef = useRef<number>(0)

  // Clock
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  // Load cameras + alert rules
  useEffect(() => {
    fetch(`${API}/cameras`).then(r => r.json()).then(setCameras).catch(() => {})
    fetchAlertRules()
  }, [])

  // WebSocket for live alerts
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
    } catch { /* ws connect fail is ok */ }
  }, [])

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  const fetchAlertRules = async () => {
    try {
      const r = await fetch(`${API}/alerts/rules`)
      setAlertRules(await r.json())
    } catch { /* ignore */ }
  }

  const doSearch = async () => {
    if (!query.trim()) return
    setSearching(true)
    try {
      const r = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, top_k: 10 }),
      })
      const data = await r.json()
      const newResults = data.results || []
      setResults(newResults)
      setSelectedResultIdx(0)
      if (newResults.length > 0) setSelectedCam(newResults[0].camera_id)
    } catch { setResults([]) }
    setSearching(false)
  }

  const selectResult = (idx: number) => {
    setSelectedResultIdx(idx)
    const result = results[idx]
    setSelectedCam(result.camera_id)
    if (mainVideoRef.current && result.timestamp >= 0) {
      mainVideoRef.current.currentTime = result.timestamp
      mainVideoRef.current.play()
    }
  }

  const registerAlert = async () => {
    if (!alertQuery.trim()) return
    await fetch(`${API}/alerts/rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
      const r = await fetch(`${API}/analytics/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: question }),
      })
      const data = await r.json()
      setChatMessages(prev => [...prev, { role: 'assistant', content: data.answer || 'No answer.' }])
    } catch {
      setChatMessages(prev => [...prev, { role: 'assistant', content: 'Error connecting to backend.' }])
    }
    setChatLoading(false)
  }

  const activeResult = results[selectedResultIdx] || null

  // Seek to detection timestamp
  useEffect(() => {
    if (activeResult && mainVideoRef.current && selectedCam === activeResult.camera_id) {
      mainVideoRef.current.currentTime = activeResult.timestamp
    }
  }, [selectedResultIdx])

  // Fetch tracking data
  useEffect(() => {
    if (selectedCam) {
      fetch(`${API}/cameras/${selectedCam}/tracks`)
        .then(r => r.json())
        .then(setTrackData)
        .catch(() => setTrackData([]))
    } else {
      setTrackData([])
    }
  }, [selectedCam])

  // Animate bounding boxes
  useEffect(() => {
    if (!trackData.length) { setLiveBboxes([]); return }
    const tracks = new Map<number, TrackDetection[]>()
    for (const d of trackData) {
      if (!tracks.has(d.track_id)) tracks.set(d.track_id, [])
      tracks.get(d.track_id)!.push(d)
    }

    const animate = () => {
      const vid = mainVideoRef.current
      if (!vid || vid.paused) { animRef.current = requestAnimationFrame(animate); return }
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
        const frac = dt > 0 ? (t - before!.timestamp) / dt : 0
        const lerp = (a: number, b: number) => a + (b - a) * Math.max(0, Math.min(1, frac))
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

  const captionPreview = activeResult?.caption
    ? activeResult.caption.length > 200
      ? activeResult.caption.slice(0, 200) + '...'
      : activeResult.caption
    : null

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] text-[#e5e5e5] overflow-hidden">

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-2.5 border-b border-[#1a1a1a] bg-[#0d0d0d]">
        <div className="flex items-center gap-3">
          <Eye size={20} className="text-[#00ff88]" />
          <span className="text-base font-semibold tracking-[0.2em] uppercase">Looking Glass</span>
          <span className="text-[10px] text-[#555] font-mono tracking-wider ml-1">SPRINGINEERING 2026</span>
        </div>
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00ff88] live-dot" />
            <span className="text-[#00ff88] font-mono text-xs font-medium">{cameras.length} LIVE</span>
          </div>
          <div className="h-4 w-px bg-[#222]" />
          <span className="font-mono text-xs text-[#666]">{clock.toLocaleTimeString()}</span>
        </div>
      </header>

      {/* ── Search Bar ── */}
      <div className="px-6 py-2.5 border-b border-[#1a1a1a] bg-[#0d0d0d]">
        <div className="flex gap-2 max-w-4xl">
          <div className="flex-1 flex items-center gap-3 bg-[#111] border border-[#222] rounded-lg px-4 py-2 focus-within:border-[#00ff88]/50 focus-within:bg-[#0f0f0f] transition-all">
            <Search size={16} className="text-[#555] flex-shrink-0" />
            <input
              className="flex-1 bg-transparent outline-none text-sm text-[#e5e5e5] placeholder-[#444]"
              placeholder='Ask anything: "find the orange truck", "person in red jacket"'
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && doSearch()}
            />
            {query && (
              <button onClick={() => setQuery('')} className="text-[#555] hover:text-[#888] text-xs">Clear</button>
            )}
          </div>
          <button
            onClick={doSearch}
            disabled={searching}
            className="px-5 py-2 bg-[#00ff88] text-[#0a0a0a] text-sm font-semibold rounded-lg hover:bg-[#00ee7d] active:bg-[#00dd72] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {searching ? (
              <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
            ) : 'Search'}
          </button>
        </div>
      </div>

      {/* ── Main Content ── */}
      <div className="flex-1 flex overflow-hidden">

        {/* ── Left: Cameras + Video ── */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Camera Grid */}
          <div className="grid grid-cols-4 gap-1.5 p-3 bg-[#0a0a0a]">
            {cameras.map(cam => (
              <div
                key={cam.camera_id}
                onClick={() => { setSelectedCam(cam.camera_id); setSelectedResultIdx(-1) }}
                className={`cam-tile relative cursor-pointer rounded-md overflow-hidden border ${
                  selectedCam === cam.camera_id
                    ? 'border-[#00ff88]/60 glow-green'
                    : 'border-[#1a1a1a] hover:border-[#333]'
                }`}
              >
                <video
                  src={`/videos/${cam.clip_name}`}
                  autoPlay loop muted playsInline
                  className="w-full aspect-video object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 px-2 py-1 flex items-center justify-between">
                  <span className="font-mono text-[10px] font-medium text-[#00ff88] tracking-wider">{cam.camera_id.toUpperCase()}</span>
                  <div className="flex items-center gap-1">
                    <div className="w-1 h-1 rounded-full bg-red-500 rec-indicator" />
                    <span className="font-mono text-[8px] text-red-400">REC</span>
                  </div>
                </div>
                {results.some(r => r.camera_id === cam.camera_id) && (
                  <div className="absolute top-1.5 right-1.5 flex items-center gap-1 bg-[#00ff88]/20 rounded-full px-1.5 py-0.5">
                    <Activity size={8} className="text-[#00ff88]" />
                    <span className="font-mono text-[8px] text-[#00ff88]">HIT</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Main Video View */}
          <div className="flex-1 relative overflow-hidden bg-[#080808] flex items-center justify-center mx-3 mb-1 rounded-lg border border-[#1a1a1a]">
            {selectedCam ? (
              <div className="relative w-full h-full flex items-center justify-center scanlines">
                <video
                  ref={mainVideoRef}
                  key={selectedCam}
                  src={`/videos/${cameras.find(c => c.camera_id === selectedCam)?.clip_name}`}
                  autoPlay loop muted playsInline
                  className="max-w-full max-h-full object-contain"
                  onLoadedMetadata={() => {
                    if (mainVideoRef.current) {
                      setVideoDims({ w: mainVideoRef.current.videoWidth, h: mainVideoRef.current.videoHeight })
                      if (activeResult && selectedCam === activeResult.camera_id && activeResult.timestamp >= 0) {
                        mainVideoRef.current.currentTime = activeResult.timestamp
                      }
                    }
                  }}
                />

                {/* Bounding Boxes */}
                {liveBboxes.map((box, i) => (
                  <div
                    key={i}
                    className="absolute bbox-corners"
                    style={{
                      left: `${(box.x1 / videoDims.w) * 100}%`,
                      top: `${(box.y1 / videoDims.h) * 100}%`,
                      width: `${((box.x2 - box.x1) / videoDims.w) * 100}%`,
                      height: `${((box.y2 - box.y1) / videoDims.h) * 100}%`,
                      transition: 'all 0.1s linear',
                      boxShadow: '0 0 8px rgba(0, 255, 136, 0.2)',
                    }}
                  >
                    <span className="bbox-corner-tr" />
                    <span className="bbox-corner-bl" />
                    <span className="absolute -top-5 left-0 font-mono text-[10px] bg-black/70 backdrop-blur-sm text-[#00ff88] px-1.5 py-0.5 rounded-sm whitespace-nowrap font-medium border border-[#00ff88]/30">
                      {box.class_name} {(box.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}

                {/* Caption overlay */}
                {captionPreview && activeResult && selectedCam === activeResult.camera_id && (
                  <div className="absolute top-3 left-3 right-3 video-gradient-top rounded-md">
                    <p className="text-[11px] text-white/80 leading-relaxed px-3 py-2 max-w-lg">
                      {captionPreview}
                    </p>
                  </div>
                )}

                {/* Match info overlay */}
                {activeResult && selectedCam === activeResult.camera_id && (
                  <div className="absolute bottom-3 left-3 flex items-center gap-3 bg-black/80 backdrop-blur-sm rounded-md px-3 py-1.5 border border-[#222]">
                    <span className="font-mono text-xs font-semibold text-[#00ff88]">{(activeResult.score * 100).toFixed(1)}%</span>
                    <div className="w-px h-3 bg-[#333]" />
                    <span className="font-mono text-[11px] text-[#888]">{activeResult.camera_id.toUpperCase()}</span>
                    <span className="font-mono text-[11px] text-[#555]">{activeResult.timestamp.toFixed(1)}s</span>
                    {activeResult.detections.length > 0 && (
                      <>
                        <div className="w-px h-3 bg-[#333]" />
                        <span className="font-mono text-[11px] text-[#666]">{activeResult.detections.length} det</span>
                      </>
                    )}
                  </div>
                )}

                {/* Camera label */}
                <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/70 rounded-md px-2 py-1 border border-[#222]">
                  <Camera size={10} className="text-[#00ff88]" />
                  <span className="font-mono text-[10px] text-[#00ff88] font-medium">{selectedCam?.toUpperCase()}</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 text-[#333]">
                <Eye size={32} />
                <span className="text-sm">Select a camera or search to begin</span>
              </div>
            )}
          </div>

          {/* Search Results Strip */}
          {results.length > 0 && (
            <div className="px-3 pb-2">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="font-mono text-[10px] text-[#555] tracking-wider uppercase">{results.length} Results</span>
                <span className="text-[10px] text-[#333]">for</span>
                <span className="font-mono text-[10px] text-[#00ff88]">"{query}"</span>
              </div>
              <div className="flex gap-1.5 overflow-x-auto pb-1">
                {results.map((r, i) => (
                  <button
                    key={i}
                    onClick={() => selectResult(i)}
                    className={`result-card flex-shrink-0 rounded-md border px-2.5 py-1.5 text-left ${
                      selectedResultIdx === i ? 'active' : 'border-[#1a1a1a]'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[11px] font-semibold text-[#ccc]">{r.camera_id.toUpperCase()}</span>
                      <span className={`font-mono text-[10px] font-medium ${selectedResultIdx === i ? 'text-[#00ff88]' : 'text-[#555]'}`}>
                        {(r.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="font-mono text-[9px] text-[#444] mt-0.5">
                      {r.timestamp.toFixed(1)}s{r.detections.length > 0 ? ` / ${r.detections.length} det` : ''}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right Sidebar ── */}
        <div className="w-72 border-l border-[#1a1a1a] flex flex-col bg-[#0c0c0c]">

          {/* Alerts Panel */}
          <div className="h-[45%] flex flex-col border-b border-[#1a1a1a]">
            <div className="px-3 py-2 flex items-center gap-2 border-b border-[#1a1a1a] bg-[#0d0d0d]">
              <Shield size={12} className="text-[#00ff88]" />
              <span className="text-[11px] font-semibold tracking-wider uppercase">Alerts</span>
              {alerts.length > 0 && (
                <span className="ml-auto font-mono text-[9px] bg-[#00ff88]/15 text-[#00ff88] px-1.5 py-0.5 rounded-full">{alerts.length}</span>
              )}
            </div>

            {/* Alert input */}
            <div className="px-3 py-2">
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-[#111] border border-[#222] rounded-md px-2 py-1 text-[11px] outline-none focus:border-[#00ff88]/40 text-[#e5e5e5] placeholder-[#444]"
                  placeholder="Alert: person with bag..."
                  value={alertQuery}
                  onChange={e => setAlertQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && registerAlert()}
                />
                <button onClick={registerAlert} className="px-2 py-1 bg-[#00ff88] text-[#0a0a0a] rounded-md text-[11px] font-semibold hover:bg-[#00ee7d] transition-colors">+</button>
              </div>
            </div>

            {/* Active Rules */}
            {alertRules.length > 0 && (
              <div className="px-3 pb-1.5">
                {alertRules.map(rule => (
                  <div key={rule.id} className="flex items-center justify-between py-0.5 group">
                    <span className="text-[10px] text-[#666] truncate flex-1 font-mono">{rule.query}</span>
                    <button
                      onClick={() => deleteAlertRule(rule.id)}
                      className="text-[#333] hover:text-red-400 ml-1 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Trash2 size={9} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Fired Alerts */}
            <div className="flex-1 overflow-y-auto px-3">
              {alerts.length === 0 && alertRules.length === 0 && (
                <p className="text-[10px] text-[#333] mt-3 text-center">Set an alert to monitor cameras</p>
              )}
              {alerts.map((a, i) => (
                <div key={i} className="alert-entry flex items-start gap-2 py-1.5 border-b border-[#151515]">
                  <Radio size={10} className="text-[#00ff88] mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] font-medium text-[#00ff88]">{a.camera_id.toUpperCase()}</span>
                      <span className="font-mono text-[9px] text-[#444]">{(a.score * 100).toFixed(0)}%</span>
                    </div>
                    <span className="text-[10px] text-[#555] truncate block">{a.query}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat Panel */}
          <div className="flex-1 flex flex-col">
            <div className="px-3 py-2 flex items-center gap-2 border-b border-[#1a1a1a] bg-[#0d0d0d]">
              <MessageSquare size={12} className="text-[#00ff88]" />
              <span className="text-[11px] font-semibold tracking-wider uppercase">Analytics</span>
            </div>

            <div className="flex-1 overflow-y-auto px-3 py-2">
              {chatMessages.length === 0 && (
                <div className="mt-1">
                  <p className="text-[9px] text-[#333] mb-2 tracking-wider uppercase">Try asking</p>
                  <div className="flex flex-col gap-1">
                    {['How many people in the lobby?', 'What color clothes on cam03?', 'Count vehicles on cam01'].map(q => (
                      <button
                        key={q}
                        onClick={() => setChatQuery(q)}
                        className="text-[11px] text-left bg-[#111] border border-[#1a1a1a] rounded-md px-2.5 py-1.5 hover:border-[#333] hover:bg-[#141414] transition-all text-[#666]"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`chat-msg mb-2 ${msg.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-lg px-2.5 py-1.5 text-[11px] leading-relaxed max-w-[95%] ${
                    msg.role === 'user'
                      ? 'bg-[#00ff88]/8 text-[#00ff88] border border-[#00ff88]/20 rounded-br-sm'
                      : 'bg-[#111] text-[#ccc] border border-[#1a1a1a] rounded-bl-sm'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="chat-msg mb-2">
                  <div className="inline-block rounded-lg px-2.5 py-1.5 text-[11px] bg-[#111] border border-[#1a1a1a] text-[#444]">
                    <span className="loading-dots"><span>.</span><span>.</span><span>.</span></span>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="px-3 py-2 border-t border-[#1a1a1a]">
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-[#111] border border-[#222] rounded-md px-2 py-1.5 text-[11px] outline-none focus:border-[#00ff88]/40 text-[#e5e5e5] placeholder-[#444]"
                  placeholder="Ask about the footage..."
                  value={chatQuery}
                  onChange={e => setChatQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && askAnalytics()}
                />
                <button
                  onClick={askAnalytics}
                  disabled={chatLoading}
                  className="px-2 py-1 bg-[#00ff88] text-[#0a0a0a] rounded-md disabled:opacity-30 hover:bg-[#00ee7d] transition-colors"
                >
                  <Send size={11} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
