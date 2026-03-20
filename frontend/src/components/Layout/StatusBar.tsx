import React from 'react'
import { useAppStore } from '../../store/useAppStore'
import { useSignatureStatusQuery } from '../../api/queries'
import { Activity, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import './Layout.css'

export const StatusBar: React.FC = () => {
  const { status, errorMessage, signatureHash, selectedPixel, selectedPixelCount, selectionMode, matchData, setStatus } = useAppStore()
  
  const { data: cacheStatus } = useSignatureStatusQuery(
    signatureHash, 
    status === 'ready' || status === 'building_cache'
  )

  React.useEffect(() => {
    if (status === 'building_cache' && cacheStatus?.status === 'ready') {
      setStatus('ready')
    }
  }, [status, cacheStatus?.status, setStatus])

  const renderStatusIcon = () => {
    if (status === 'error') return <AlertCircle size={16} className="text-error" />
    if (status === 'loading_image' || status === 'building_cache') return <Loader size={16} className="text-accent spin" />
    return <CheckCircle size={16} className="text-success" />
  }

  const renderStatusText = () => {
    if (status === 'error') return errorMessage || 'Error occurred'
    if (status === 'loading_image') return 'Loading image and parsing metadata...'
    if (cacheStatus?.status === 'building') return `Building signature cache... ${cacheStatus.progress}% (${cacheStatus.current_step})`
    if (status === 'building_cache') return 'Signature cache preparing...'
    if (status === 'ready') return 'Ready. Click on the image to extract spectrum.'
    return 'Idle'
  }

  const renderSelectionText = () => {
    if (!selectedPixel) return 'No selection'
    const matchedMode = String(matchData?.query?.selection_mode || selectionMode)
    const mode = matchedMode === 'pixel' ? 'Pixel' : matchedMode === 'box' ? 'Box' : matchedMode === 'circle' ? 'Circle' : 'Lasso'
    const count = Number(matchData?.query?.selected_pixels || selectedPixelCount || 1)
    const countText = mode === 'Pixel' ? '' : ` | ${count} px`
    return `${mode}${countText} | X: ${selectedPixel.x}, Y: ${selectedPixel.y}`
  }

  return (
    <div className="status-bar">
      <div className="status-indicator">
        {renderStatusIcon()}
        <span>{renderStatusText()}</span>
      </div>
      <div className="status-details">
        {renderSelectionText()}
        <Activity size={16} className="ml-2" />
      </div>
    </div>
  )
}
