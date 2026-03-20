import React, { useState } from 'react'
import { useLoadImageMutation } from '../../api/queries'
import { useAppStore } from '../../store/useAppStore'
import { Upload, Clock3, X } from 'lucide-react'
import './ImageLoader.css'

type ImportHistoryItem = {
  path: string;
  displayMode: string;
  updatedAt: number;
}

const HISTORY_KEY = 'spectral_match_import_history_v1'
const HISTORY_LIMIT = 12

const normalizePathKey = (value: string) =>
  value.trim().replace(/\\/g, '/').toLowerCase()

const readImportHistory = (): ImportHistoryItem[] => {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    const deduped: ImportHistoryItem[] = []
    const seen = new Set<string>()
    for (const item of parsed) {
      if (!item || typeof item.path !== 'string' || typeof item.displayMode !== 'string') continue
      const path = String(item.path).trim()
      if (!path) continue
      const key = normalizePathKey(path)
      if (seen.has(key)) continue
      seen.add(key)
      deduped.push({
        path,
        displayMode: String(item.displayMode),
        updatedAt: Number(item.updatedAt || Date.now())
      })
      if (deduped.length >= HISTORY_LIMIT) break
    }
    return deduped
  } catch {
    return []
  }
}

export const ImageLoader: React.FC = () => {
  const [path, setPath] = useState('')
  const [displayMode, setDisplayMode] = useState('true_color')
  const [history, setHistory] = useState<ImportHistoryItem[]>(() => readImportHistory())
  
  const { mutate: loadImage, isPending } = useLoadImageMutation()
  const { setImageData, setStatus, matchOptions } = useAppStore()

  const persistHistory = (items: ImportHistoryItem[]) => {
    setHistory(items)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items))
  }

  const addHistory = (nextPath: string, nextDisplayMode: string) => {
    const normalized = nextPath.trim()
    if (!normalized) return
    const key = normalizePathKey(normalized)
    const dedup = history.filter(
      (item) => normalizePathKey(item.path) !== key
    )
    const updated: ImportHistoryItem[] = [
      { path: normalized, displayMode: nextDisplayMode, updatedAt: Date.now() },
      ...dedup
    ].slice(0, HISTORY_LIMIT)
    persistHistory(updated)
  }

  const handleLoad = () => {
    if (!path.trim()) return
    
    setStatus('loading_image')
    loadImage(
      { 
        image_path: path, 
        display_mode: displayMode,
        build_signature_cache: true,
        ignore_water_bands: matchOptions.ignoreWaterBands
      },
      {
        onSuccess: (data) => {
          addHistory(path, displayMode)
          setImageData({
            imageId: data.image_id,
            path: data.image_path,
            meta: {
              samples: data.samples,
              lines: data.lines,
              bands: data.bands,
              wavelengths: data.wavelengths
            },
            signatureHash: data.signature?.hash || null
          })
          if (data.signature?.build_status === 'building') {
            setStatus('building_cache')
          } else {
            setStatus('ready')
          }
        },
        onError: (err: any) => {
          setStatus('error', err.message || 'Failed to load image')
        }
      }
    )
  }

  return (
    <div className="image-loader">
      <div className="option-group">
        <label>Absolute File Path</label>
        <input 
          type="text" 
          className="input-text" 
          placeholder="E.g. D:/data/hsi/scene01.hdr"
          list="image-import-history"
          value={path}
          onChange={e => setPath(e.target.value)}
        />
        <datalist id="image-import-history">
          {history.map((item) => (
            <option key={`${item.path}|${item.displayMode}`} value={item.path} />
          ))}
        </datalist>
      </div>
      
      <div className="option-group">
        <label>Display Mode</label>
        <select value={displayMode} onChange={e => setDisplayMode(e.target.value)}>
          <option value="true_color">True Color (RGB)</option>
          <option value="false_color">False Color (NIR,R,G)</option>
        </select>
      </div>

      <button 
        className="btn w-full mt-2" 
        onClick={handleLoad} 
        disabled={isPending || !path}
      >
        <Upload size={16} />
        {isPending ? 'Loading...' : 'Load Image'}
      </button>

      {history.length > 0 && (
        <div className="import-history">
          <div className="import-history-header">
            <div className="import-history-title">
              <Clock3 size={14} />
              <span>Recent Imports</span>
            </div>
            <button
              type="button"
              className="import-history-clear"
              onClick={() => {
                persistHistory([])
                setPath('')
              }}
            >
              <X size={13} />
              Clear
            </button>
          </div>
          <div className="import-history-list">
            {history.slice(0, 5).map((item) => (
              <button
                key={`${item.path}|${item.displayMode}`}
                type="button"
                className="import-history-item"
                onClick={() => {
                  setPath(item.path)
                  setDisplayMode(item.displayMode)
                }}
                title={`${item.path} (${item.displayMode})`}
              >
                <span className="import-history-mode">
                  {item.displayMode === 'false_color' ? 'False Color' : 'True Color'}
                </span>
                <span className="import-history-path">{item.path}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
