import { useState, useEffect, useRef } from 'react'
import { Search, Bell, MessageSquare, Radio } from 'lucide-react'
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

interface AlertEvent {
  rule_id: string
  query: string
  camera_id: string
  timestamp: number
  score: number
}

function App() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResultItem[]>([])
  const [searching, setSearching] = useState(false)
  const [selectedCam, setSelectedCam] = useState<string | null>(null)
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [alertQuery, setAlertQuery] = useState('')
  const [chatQuery, setChatQuery] = useState('')
  const [chatAnswer, setChatAnswer] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [clock, setClock] = useState(new Date())
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    fetch(`${API}/cameras`).then(r => r.json()).then(setCameras).catch(() => {})
  }, [])

  useEffect(() => {
    try {
      const ws = new WebSocket(`ws://localhost:8000/alerts/ws`)
      wsRef.current = ws
      ws.onmessage = (e) => {
        const alert = JSON.parse(e.data) as AlertEvent
        setAlerts(prev => [alert, ...prev].slice(0, 20))
      }
      const hb = setInterval(() => { if (ws.readyState === 1) ws.send('ping') }, 30000)
      return () => { clearInterval(hb); ws.close() }
    } catch { /* ws connect fail is ok */ }
  }, [])

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
      setResults(data.results || [])
      if (data.results?.length > 0) setSelectedCam(data.results[0].camera_id)
    } catch { setResults([]) }
    setSearching(false)
  }

  const registerAlert = async () => {
    if (!alertQuery.trim()) return
    await fetch(`${API}/alerts/rules`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: alertQuery }),
    })
    setAlertQuery('')
  }

  const askAnalytics = async () => {
    if (!chatQuery.trim()) return
    setChatLoading(true)
    try {
      const r = await fetch(`${API}/analytics/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: chatQuery }),
      })
      const data = await r.json()
      setChatAnswer(data.answer || 'No answer.')
    } catch { setChatAnswer('Error connecting to backend.') }
    setChatLoading(false)
  }

  const topResult = results[0]

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
              placeholder='Ask anything: "find the orange truck"'
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
            {searching ? '...' : 'Search'}
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
                onClick={() => setSelectedCam(cam.camera_id)}
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

          <div className="flex-1 relative rounded-lg overflow-hidden border border-[#2a2a2a] bg-[#141414] scanlines">
            {selectedCam && (
              <video
                key={selectedCam}
                src={`/videos/${cameras.find(c => c.camera_id === selectedCam)?.clip_name}`}
                autoPlay loop muted playsInline
                className="w-full h-full object-contain"
              />
            )}
            {topResult && selectedCam === topResult.camera_id && topResult.detections?.map((det, i) => (
              det.bbox && (
                <div
                  key={i}
                  className="absolute border-2 border-[#00ff88] bbox-animate"
                  style={{
                    left: `${(det.bbox[0] / 1280) * 100}%`,
                    top: `${(det.bbox[1] / 720) * 100}%`,
                    width: `${((det.bbox[2] - det.bbox[0]) / 1280) * 100}%`,
                    height: `${((det.bbox[3] - det.bbox[1]) / 720) * 100}%`,
                  }}
                >
                  <span className="absolute -top-5 left-0 font-mono text-xs bg-[#00ff88] text-black px-1 rounded">
                    {det.class_name} {(det.score * 100).toFixed(0)}%
                  </span>
                </div>
              )
            ))}
            {topResult && selectedCam === topResult.camera_id && (
              <div className="absolute bottom-4 left-4 bg-black/80 rounded-lg px-4 py-2 font-mono text-sm">
                <span className="text-[#00ff88]">Match: {(topResult.score * 100).toFixed(1)}%</span>
                <span className="text-[#888] ml-3">{topResult.camera_id} · {topResult.timestamp.toFixed(1)}s</span>
              </div>
            )}
            {topResult && selectedCam === topResult.camera_id && topResult.caption && (
              <div className="absolute top-4 left-4 bg-black/80 rounded-lg px-3 py-1 text-xs text-[#888] max-w-md">
                {topResult.caption}
              </div>
            )}
            {!selectedCam && (
              <div className="absolute inset-0 flex items-center justify-center text-[#555]">
                Select a camera or search
              </div>
            )}
          </div>

          {results.length > 0 && (
            <div className="flex gap-2 mt-2 overflow-x-auto pb-2">
              {results.map((r, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedCam(r.camera_id)}
                  className={`flex-shrink-0 rounded border px-3 py-1 text-xs font-mono transition-colors ${
                    selectedCam === r.camera_id ? 'border-[#00ff88] text-[#00ff88]' : 'border-[#2a2a2a] text-[#888] hover:border-[#555]'
                  }`}
                >
                  {r.camera_id} · {(r.score * 100).toFixed(1)}%
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right sidebar */}
        <div className="w-80 border-l border-[#2a2a2a] flex flex-col overflow-hidden">
          {/* Alerts */}
          <div className="flex-1 flex flex-col border-b border-[#2a2a2a] overflow-hidden">
            <div className="px-4 py-2 flex items-center gap-2 border-b border-[#2a2a2a]">
              <Bell size={14} className="text-[#00ff88]" />
              <span className="text-sm font-semibold">ALERTS</span>
            </div>
            <div className="px-4 py-2">
              <div className="flex gap-1">
                <input
                  className="flex-1 bg-[#141414] border border-[#2a2a2a] rounded px-2 py-1 text-xs outline-none focus:border-[#00ff88] text-[#e5e5e5]"
                  placeholder="Register alert..."
                  value={alertQuery}
                  onChange={e => setAlertQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && registerAlert()}
                />
                <button onClick={registerAlert} className="text-xs px-2 py-1 bg-[#00ff88] text-black rounded font-semibold">+</button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-4">
              {alerts.length === 0 && <p className="text-xs text-[#555] mt-2">No alerts yet.</p>}
              {alerts.map((a, i) => (
                <div key={i} className="flex items-start gap-2 py-1.5 border-b border-[#1a1a1a] text-xs">
                  <Radio size={12} className="text-[#00ff88] mt-0.5 flex-shrink-0" />
                  <div>
                    <span className="text-[#00ff88] font-mono">{a.camera_id}</span>
                    <span className="text-[#888] ml-2">{a.query}</span>
                    <span className="text-[#555] ml-1 font-mono">{(a.score * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Chat */}
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="px-4 py-2 flex items-center gap-2 border-b border-[#2a2a2a]">
              <MessageSquare size={14} className="text-[#00ff88]" />
              <span className="text-sm font-semibold">ANALYTICS CHAT</span>
            </div>
            <div className="flex-1 overflow-y-auto px-4 py-2">
              <div className="flex flex-wrap gap-1 mb-3">
                {['How many people in the lobby?', 'Count vehicles on cam01'].map(q => (
                  <button
                    key={q}
                    onClick={() => setChatQuery(q)}
                    className="text-xs bg-[#141414] border border-[#2a2a2a] rounded-full px-3 py-1 hover:border-[#00ff88] transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
              {chatAnswer && (
                <div className="bg-[#141414] rounded-lg p-3 text-sm">
                  <p className="text-[#00ff88] font-mono text-xs mb-1">Answer:</p>
                  <p>{chatAnswer}</p>
                </div>
              )}
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
                  className="text-xs px-3 py-1 bg-[#00ff88] text-black rounded font-semibold disabled:opacity-50"
                >
                  {chatLoading ? '...' : 'Ask'}
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
