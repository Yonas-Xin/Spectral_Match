import React from 'react'
import { useAppStore } from '../../store/useAppStore'
import { useExportResultMutation } from '../../api/queries'
import { Download, Check } from 'lucide-react'
import './MatchTable.css'

export const MatchTable: React.FC = () => {
  const { matchData, imageId, selectedPixel, selection, matchOptions } = useAppStore()
  const [format, setFormat] = React.useState('csv')
  const [outputPath, setOutputPath] = React.useState('')
  const [showSuccess, setShowSuccess] = React.useState(false)
  const { mutate: exportResult, isPending: exporting } = useExportResultMutation()

  if (!matchData || !matchData.results) {
    return (
      <div className="match-table-container panel">
        <div className="panel-header">
          <h3>Match Results</h3>
        </div>
        <div className="empty-state p-4">
          No matches yet. Select a pixel to view top N results.
        </div>
      </div>
    )
  }

  const { results, match_context } = matchData
  const canExport = !!imageId && !!selectedPixel && !!outputPath.trim()

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
          setTimeout(() => setShowSuccess(false), 2500)
        }
      }
    )
  }

  return (
    <div className="match-table-container panel">
      <div className="panel-header">
        <h3>Top {match_context.candidate_count > 0 ? results.length : 0} Matches</h3>
        <span className="text-sm text-accent">
          {match_context.metric.toUpperCase()} | {match_context.elapsed_ms}ms
        </span>
      </div>

      <div className="match-export-bar">
        <div className="match-export-fields">
          <select value={format} onChange={(e) => setFormat(e.target.value)} className="match-export-select">
            <option value="csv">CSV</option>
            <option value="txt">TXT</option>
          </select>
          <input
            className="match-export-path"
            type="text"
            placeholder="D:/output/match_curves.csv"
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
          />
        </div>
        <button className="btn btn-secondary match-export-btn" onClick={handleExport} disabled={!canExport || exporting}>
          {showSuccess ? <Check size={14} className="text-success" /> : <Download size={14} />}
          {exporting ? 'Exporting...' : showSuccess ? 'Exported' : 'Export Curves'}
        </button>
      </div>

      <div className="table-wrapper">
        <table className="match-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Class</th>
              <th>Name</th>
              <th>SAM Score</th>
              <th>Pearson (r)</th>
            </tr>
          </thead>
          <tbody>
            {results.map((res: any) => (
              <tr key={res.rank}>
                <td><span className="rank-badge">{res.rank}</span></td>
                <td className="class-chip-cell">
                  <span className="class-chip">{res.class || 'Unknown'}</span>
                </td>
                <td className="font-medium text-primary">{res.name}</td>
                <td className="font-mono">{res.sam_score?.toFixed(4)}</td>
                <td className="font-mono">{res.pearson_r?.toFixed(4) ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
