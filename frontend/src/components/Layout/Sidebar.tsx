import React from 'react'
import { ImageLoader } from '../ImageLoader'
import { useAppStore } from '../../store/useAppStore'
import { ChevronDown, ChevronUp, FileImage, SlidersHorizontal, Wrench } from 'lucide-react'
import './Layout.css'

export const Sidebar: React.FC = () => {
  const { matchOptions, setMatchOptions } = useAppStore()
  const [collapsed, setCollapsed] = React.useState({
    workspace: false,
    options: false,
    utilities: false
  })

  const toggleCard = (key: 'workspace' | 'options' | 'utilities') => {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  return (
    <aside className="sidebar">
      <div className="dock-card dock-brand-card">
        <div className="dock-card-header brand-card-header">
          <div className="dock-title">
            <FileImage size={18} className="icon-accent" />
            <h3>Spectral Match</h3>
          </div>
          <span className="brand-pill">Dock</span>
        </div>
      </div>

      <div className="dock-card">
        <div className="dock-card-header">
          <div className="dock-title">
            <FileImage size={16} />
            <h3>Image Workspace</h3>
          </div>
          <button className="dock-toggle" onClick={() => toggleCard('workspace')}>
            {collapsed.workspace ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
        {!collapsed.workspace && (
          <div className="dock-card-body">
            <ImageLoader />
          </div>
        )}
      </div>

      <div className="dock-card">
        <div className="dock-card-header">
          <div className="dock-title">
            <SlidersHorizontal size={16} />
            <h3>Match Options</h3>
          </div>
          <button className="dock-toggle" onClick={() => toggleCard('options')}>
            {collapsed.options ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
        {!collapsed.options && (
          <div className="dock-card-body">
            <div className="option-group">
              <label>Top N Matches</label>
              <select
                value={matchOptions.topN}
                onChange={e => setMatchOptions({ topN: Number(e.target.value) })}
              >
                <option value="5">5</option>
                <option value="10">10</option>
                <option value="20">20</option>
              </select>
            </div>
            <div className="option-group">
              <label>Metric</label>
              <select
                value={matchOptions.metric}
                onChange={e => setMatchOptions({ metric: e.target.value })}
              >
                <option value="sam">SAM (Spectral Angle)</option>
              </select>
            </div>
            <div className="option-group">
              <label>Min Valid Bands</label>
              <input
                type="number"
                className="input-text"
                min={1}
                step={1}
                value={matchOptions.minValidBands}
                onChange={(e) => {
                  const n = Number(e.target.value)
                  if (!Number.isFinite(n)) return
                  setMatchOptions({ minValidBands: Math.max(1, Math.round(n)) })
                }}
              />
            </div>
            <div className="option-group checkbox-group">
              <input
                type="checkbox"
                id="waterBands"
                checked={matchOptions.ignoreWaterBands}
                onChange={e => setMatchOptions({ ignoreWaterBands: e.target.checked })}
              />
              <label htmlFor="waterBands">Ignore Water Vapor Bands</label>
            </div>
            <div className="option-group checkbox-group">
              <input
                type="checkbox"
                id="showWaterRanges"
                checked={matchOptions.showWaterBandRanges}
                onChange={e => setMatchOptions({ showWaterBandRanges: e.target.checked })}
              />
              <label htmlFor="showWaterRanges">Show Water Band Ranges</label>
            </div>
          </div>
        )}
      </div>

      <div className="dock-card">
        <div className="dock-card-header">
          <div className="dock-title">
            <Wrench size={16} />
            <h3>Utilities</h3>
          </div>
          <button className="dock-toggle" onClick={() => toggleCard('utilities')}>
            {collapsed.utilities ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
          </button>
        </div>
        {!collapsed.utilities && (
          <div className="dock-card-body">
            <div className="dock-note">
              Curve export has been moved to the top area of <strong>Match Results</strong> for faster workflow.
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}
