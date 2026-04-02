import { useState, useEffect, useRef } from 'react'
import { Search, Bell, MessageSquare, Radio, Trash2, ChevronRight } from 'lucide-react'
import './index.css'

const API = '/api'

interface Camera {
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
  const [cameras, setCameras] = useState<Camera[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResultItem[]>([])
  const [selectedResultIdx, setSelectedResultIdx] = useState<number>(0)
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
    } catch { /* ws connect fail is ok */ }
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages])

  const fetchAlertRules = async () => {
    try {
      const r = await fetch(`${API}/alerts/rules`)
      const data = await r.json()
      setAlertRules(data)
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
    // Seek video to the detection timestamp, then keep playing
    if (mainVideoRef.current && result.timestamp >= 0) {
      mainVideoRef.current.currentTime = result.timestamp
      mainVideoRef.current.play()
    }
  }

  const registerAlert = async () => {
    if (!alertQuery.trim()) return
    try {
      await fetch(`${API}/alerts/rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: alertQuery }),
      })
      setAlertQuery('')
      fetchAlertRules()
    } catch { /* ignore */ }
  }

  const deleteAlertRule = async (ruleId: string) => {
    try {
      await fetch(`${API}/alerts/rules/${ruleId}`, { method: 'DELETE' })
      fetchAlertRules()
    } catch { /* ignore */ }
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

  // Seek to detection timestamp when switching between results on the same camera
  useEffect(() => {
    if (activeResult && mainVideoRef.current && selectedCam === activeResult.camera_id) {
      mainVideoRef.current.currentTime = activeResult.timestamp
    }
  }, [selectedResultIdx])

  // Fetch tracking data when camera changes
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

  // Animate bounding boxes by interpolating tracking data at current video time
  useEffect(() => {
    if (!trackData.length) { setLiveBboxes([]); return }

    const animate = () => {
      const vid = mainVideoRef.current
      if (!vid || vid.paused) { animRef.current = requestAnimationFrame(animate); return }
      const t = vid.currentTime

      // Group detections by track_id
      const tracks = new Map<number, TrackDetection[]>()
      for (const d of trackData) {
        if (!tracks.has(d.track_id)) tracks.set(d.track_id, [])
        tracks.get(d.track_id)!.push(d)
      }

      const boxes: typeof liveBboxes = []
      for (const [, dets] of tracks) {
        // Find the two detections surrounding current time for interpolation
        let before: TrackDetection | null = null
        let after: TrackDetection | null = null
        for (const d of dets) {
          if (d.timestamp <= t) before = d
          if (d.timestamp >= t && !after) after = d
        }
        if (!before && !after) continue
        if (before && !after) after = before
        if (!before && after) before = after

        // Linear interpolation
        const dt = after!.timestamp - before!.timestamp
        const frac = dt > 0 ? (t - before!.timestamp) / dt : 0
        const lerp = (a: number, b: number) => a + (b - a) * Math.max(0, Math.min(1, frac))

        // Only show if current time is within the track's active range (with 0.5s margin)
        if (t < before!.timestamp - 0.5 || t > after!.timestamp + 0.5) continue

        boxes.push({
          x1: lerp(before!.x1, after!.x1),
          y1: lerp(before!.y1, after!.y1),
          x2: lerp(before!.x2, after!.x2),
          y2: lerp(before!.y2, after!.y2),
          class_name: before!.class_name,
          score: before!.score,
        })
      }
      setLiveBboxes(boxes)
      animRef.current = requestAnimationFrame(animate)
    }

    animRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animRef.current)
  }, [trackData])

  return (
    <div className="h-screen flex flex-col bg-[#0a0a0a] text-[#e5e5e5] overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-[#2a2a2a]">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
          <span className="text-lg font-semibold tracking-wider">LOOKING GLASS</span>
          <span className="text-xs text-[#888]">Springineering 2026</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-[#00ff88] font-mono text-sm">{cameras.length} cameras</span>
          <span className="font-mono text-sm text-[#888]">{clock.toLocaleTimeString()}</span>
        </div>
      </header>

      {/* Search bar */}
      <div className="px-6 py-3 border-b border-[#2a2a2a]">
        <div className="flex gap-2 max-w-3xl">
          <div className="flex-1 flex items-center gap-2 bg-[#141414] border border-[#2a2a2a] rounded-lg px-4 py-2 focus-within:border-[#00ff88] transition-colors">
            <Search size={18} className="text-[#888]" />
            <input
              className="flex-1 bg-transparent outline-none text-[#e5e5e5] placeholder-[#555]"
              placeholder='Search across all cameras: "find the orange truck"'
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && doSearch()}
            />
          </div>
          <button
            onClick={doSearch}
            disabled={searching}
            className="px-6 py-2 bg-[#00ff88] text-black font-semibold rounded-lg hover:bg-[#00dd77] transition-colors disabled:opacity-50"
          >
            {searching ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Camera grid + selected view */}
        <div className="flex-1 flex flex-col overflow-hidden p-4">
          <div className="grid grid-cols-4 gap-2 mb-4">
            {cameras.map(cam => (
              <div
                key={cam.camera_id}
                onClick={() => { setSelectedCam(cam.camera_id); setSelectedResultIdx(-1) }}
                className={`relative cursor-pointer rounded overflow-hidden border transition-all hover:border-[#00ff88] ${
                  selectedCam === cam.camera_id ? 'border-[#00ff88] glow-green' : 'border-[#2a2a2a]'
                }`}
              >
                <video
                  src={`/videos/${cam.clip_name}`}
                  autoPlay loop muted playsInline
                  className="w-full h-24 object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-2 py-0.5 text-xs font-mono text-[#00ff88]">
                  {cam.camera_id}
                </div>
                {results.some(r => r.camera_id === cam.camera_id) && (
                  <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-[#00ff88] animate-pulse" />
                )}
              </div>
            ))}
          </div>

          <div className="flex-1 relative rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#141414] scanlines flex items-center justify-center">
            {selectedCam && (
              <div className="relative" style={{ aspectRatio: `${videoDims.w} / ${videoDims.h}`, maxWidth: '100%', maxHeight: '100%' }}>
                <video
                  ref={mainVideoRef}
                  key={selectedCam}
                  src={`/videos/${cameras.find(c => c.camera_id === selectedCam)?.clip_name}`}
                  autoPlay loop muted playsInline
                  className="w-full h-full"
                  onLoadedMetadata={() => {
                    if (mainVideoRef.current) {
                      setVideoDims({ w: mainVideoRef.current.videoWidth, h: mainVideoRef.current.videoHeight })
                      // Seek to detection timestamp when video loads for a search result
                      if (activeResult && selectedCam === activeResult.camera_id && activeResult.timestamp >= 0) {
                        mainVideoRef.current.currentTime = activeResult.timestamp
                      }
                    }
                  }}
                />
                {liveBboxes.map((box, i) => (
                  <div
                    key={i}
                    className="absolute border-2 border-[#00ff88]"
                    style={{
                      left: `${(box.x1 / videoDims.w) * 100}%`,
                      top: `${(box.y1 / videoDims.h) * 100}%`,
                      width: `${((box.x2 - box.x1) / videoDims.w) * 100}%`,
                      height: `${((box.y2 - box.y1) / videoDims.h) * 100}%`,
                      transition: 'all 0.1s linear',
                    }}
                  >
                    <span className="absolute -top-5 left-0 font-mono text-xs bg-[#00ff88] text-black px-1 rounded whitespace-nowrap">
                      {box.class_name} {(box.score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
                {activeResult && selectedCam === activeResult.camera_id && (
                  <div className="absolute bottom-4 left-4 bg-black/80 rounded-lg px-4 py-2 font-mono text-sm">
                    <span className="text-[#00ff88]">Match: {(activeResult.score * 100).toFixed(1)}%</span>
                    <span className="text-[#888] ml-3">{activeResult.camera_id} · {activeResult.timestamp.toFixed(1)}s</span>
                    {activeResult.detections.length > 0 && (
                      <span className="text-[#555] ml-3">{activeResult.detections.length} detection{activeResult.detections.length > 1 ? 's' : ''}</span>
                    )}
                  </div>
                )}
                {activeResult && selectedCam === activeResult.camera_id && activeResult.caption && (
                  <div className="absolute top-4 left-4 bg-black/80 rounded-lg px-3 py-1 text-xs text-[#aaa] max-w-md leading-relaxed">
                    {activeResult.caption}
                  </div>
                )}
              </div>
            )}
            {!selectedCam && (
              <div className="absolute inset-0 flex items-center justify-center text-[#555]">
                Select a camera or search to begin
              </div>
            )}
          </div>

          {/* Search results strip */}
          {results.length > 0 && (
            <div className="mt-3">
              <div className="text-xs text-[#888] mb-1 font-mono">
                {results.length} results for "{query}"
              </div>
              <div className="flex gap-2 overflow-x-auto pb-2">
                {results.map((r, i) => (
                  <button
                    key={i}
                    onClick={() => selectResult(i)}
                    className={`flex-shrink-0 rounded-lg border px-3 py-2 text-xs font-mono transition-all ${
                      selectedResultIdx === i
                        ? 'border-[#00ff88] bg-[#00ff88]/10 text-[#00ff88]'
                        : 'border-[#2a2a2a] text-[#888] hover:border-[#555] hover:bg-[#141414]'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">{r.camera_id}</span>
                      <span className={selectedResultIdx === i ? 'text-[#00ff88]' : 'text-[#555]'}>{(r.score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-[10px] text-[#555] mt-0.5 text-left">
                      {r.timestamp.toFixed(1)}s
                      {r.detections.length > 0 && ` · ${r.detections.length} det`}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="w-80 border-l border-[#2a2a2a] flex flex-col overflow-hidden">
          {/* Alerts */}
          <div className="h-1/2 flex flex-col border-b border-[#2a2a2a] overflow-hidden">
            <div className="px-4 py-2 flex items-center gap-2 border-b border-[#2a2a2a]">
              <Bell size={14} className="text-[#00ff88]" />
              <span className="text-sm font-semibold">ALERTS</span>
              {alerts.length > 0 && (
                <span className="ml-auto text-[10px] bg-[#00ff88]/20 text-[#00ff88] px-2 py-0.5 rounded-full font-mono">
                  {alerts.length}
                </span>
              )}
            </div>
            <div className="px-4 py-2">
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-[#141414] border border-[#2a2a2a] rounded px-2 py-1 text-xs outline-none focus:border-[#00ff88] text-[#e5e5e5]"
                  placeholder="Register alert rule..."
                  value={alertQuery}
                  onChange={e => setAlertQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && registerAlert()}
                />
                <button onClick={registerAlert} className="text-xs px-2 py-1 bg-[#00ff88] text-black rounded font-semibold">+</button>
              </div>
            </div>

            {/* Active rules */}
            {alertRules.length > 0 && (
              <div className="px-4 pb-2">
                <div className="text-[10px] text-[#555] mb-1 uppercase tracking-wider">Active Rules</div>
                {alertRules.map(rule => (
                  <div key={rule.id} className="flex items-center justify-between py-1 text-xs">
                    <span className="text-[#aaa] truncate flex-1">{rule.query}</span>
                    <button
                      onClick={() => deleteAlertRule(rule.id)}
                      className="text-[#555] hover:text-red-400 ml-2 flex-shrink-0"
                    >
                      <Trash2 size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Fired alerts */}
            <div className="flex-1 overflow-y-auto px-4">
              {alerts.length === 0 && alertRules.length === 0 && (
                <p className="text-xs text-[#555] mt-2">Register an alert rule to get notified when matching objects appear.</p>
              )}
              {alerts.map((a, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b border-[#1a1a1a] text-xs">
                  <Radio size={12} className="text-[#00ff88] mt-0.5 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1">
                      <span className="text-[#00ff88] font-mono">{a.camera_id}</span>
                      <span className="text-[#555] font-mono">{(a.score * 100).toFixed(0)}%</span>
                    </div>
                    <span className="text-[#888] truncate block">{a.query}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="h-1/2 flex flex-col overflow-hidden">
            <div className="px-4 py-2 flex items-center gap-2 border-b border-[#2a2a2a]">
              <MessageSquare size={14} className="text-[#00ff88]" />
              <span className="text-sm font-semibold">ANALYTICS CHAT</span>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-2">
              {chatMessages.length === 0 && (
                <div className="mb-3">
                  <p className="text-[10px] text-[#555] mb-2 uppercase tracking-wider">Suggested questions</p>
                  <div className="flex flex-wrap gap-1">
                    {['How many people in the lobby?', 'Count vehicles on cam01', 'What objects are on cam06?'].map(q => (
                      <button
                        key={q}
                        onClick={() => setChatQuery(q)}
                        className="text-xs bg-[#141414] border border-[#2a2a2a] rounded-full px-3 py-1 hover:border-[#00ff88] transition-colors text-left"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`mb-2 ${msg.role === 'user' ? 'text-right' : ''}`}>
                  <div className={`inline-block rounded-lg px-3 py-2 text-sm max-w-[90%] ${
                    msg.role === 'user'
                      ? 'bg-[#00ff88]/10 text-[#00ff88] border border-[#00ff88]/30'
                      : 'bg-[#141414] text-[#e5e5e5] border border-[#2a2a2a]'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="mb-2">
                  <div className="inline-block rounded-lg px-3 py-2 text-sm bg-[#141414] border border-[#2a2a2a] text-[#555]">
                    Analyzing footage...
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <div className="px-4 py-2 border-t border-[#2a2a2a]">
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-[#141414] border border-[#2a2a2a] rounded px-2 py-1 text-xs outline-none focus:border-[#00ff88] text-[#e5e5e5]"
                  placeholder="Ask about the footage..."
                  value={chatQuery}
                  onChange={e => setChatQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && askAnalytics()}
                />
                <button
                  onClick={askAnalytics}
                  disabled={chatLoading}
                  className="text-xs px-2 py-1 bg-[#00ff88] text-black rounded font-semibold disabled:opacity-50 flex items-center gap-1"
                >
                  <ChevronRight size={12} />
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
