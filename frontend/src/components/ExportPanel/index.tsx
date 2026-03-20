import React, { useState } from 'react'
import { useExportResultMutation } from '../../api/queries'
import { useAppStore } from '../../store/useAppStore'
import { Download, Check } from 'lucide-react'
import './ExportPanel.css'

export const ExportPanel: React.FC = () => {
  const [format, setFormat] = useState('csv')
  const [outputPath, setOutputPath] = useState('')
  const [showSuccess, setShowSuccess] = useState(false)
  
  const { imageId, selectedPixel, selection, matchOptions } = useAppStore()
  const { mutate: exportResult, isPending } = useExportResultMutation()

  const handleExport = () => {
    if (!imageId || !selectedPixel || !outputPath.trim()) return

    exportResult(
      {
        image_id: imageId,
        x: selectedPixel.x,
        y: selectedPixel.y,
        top_n: matchOptions.topN,
        format,
        output_path: outputPath,
        include_query_spectrum: true,
        include_matched_curves: true,
        ignore_water_bands: matchOptions.ignoreWaterBands,
        min_valid_bands: matchOptions.minValidBands,
        selection: selection ?? undefined
      },
      {
        onSuccess: () => {
          setShowSuccess(true)
          setTimeout(() => setShowSuccess(false), 3000)
        },
        onError: (err: any) => {
          alert('Export failed: ' + err.message)
        }
      }
    )
  }

  const isReady = !!imageId && !!selectedPixel

  return (
    <div className="export-panel">
      {!isReady && (
        <div className="empty-state-text mb-2">
          Select a pixel first to enable export.
        </div>
      )}
      <div className={`export-controls ${!isReady ? 'disabled' : ''}`}>
        <div className="option-group">
          <label>Format</label>
          <select value={format} onChange={e => setFormat(e.target.value)} disabled={!isReady}>
            <option value="csv">CSV</option>
            <option value="txt">TXT</option>
          </select>
        </div>
        
        <div className="option-group">
          <label>Output Absolute Path</label>
          <input 
            type="text" 
            className="input-text" 
            placeholder="D:/output/result.csv"
            value={outputPath}
            onChange={e => setOutputPath(e.target.value)}
            disabled={!isReady}
          />
        </div>

        <button 
          className="btn btn-secondary w-full mt-2" 
          onClick={handleExport} 
          disabled={!isReady || isPending || !outputPath}
        >
          {showSuccess ? <Check size={16} className="text-success" /> : <Download size={16} />}
          {isPending ? 'Exporting...' : showSuccess ? 'Exported!' : 'Export Results'}
        </button>
      </div>
    </div>
  )
}
