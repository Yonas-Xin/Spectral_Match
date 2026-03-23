import React from 'react'
import { Search, X } from 'lucide-react'
import { useLibraryCurvesQuery, useLibrarySearchQuery } from '../../api/queries'
import { useAppStore } from '../../store/useAppStore'
import type { LibrarySearchItem } from '../../types/spectral'
import './LibrarySearchPanel.css'

const SEARCH_LIMIT = 18

export const LibrarySearchPanel: React.FC = () => {
  const {
    imageId,
    selectedLibrarySpectra,
    toggleSelectedLibrarySpectrum,
    addSelectedLibrarySpectra,
    removeSelectedLibrarySpectra,
    clearSelectedLibrarySpectra,
    setSelectedLibraryCurves
  } = useAppStore()
  const [queryInput, setQueryInput] = React.useState('')
  const [debouncedQuery, setDebouncedQuery] = React.useState('')

  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedQuery(queryInput.trim())
    }, 220)
    return () => window.clearTimeout(timer)
  }, [queryInput])

  const selectedIds = React.useMemo(
    () => selectedLibrarySpectra.map((item) => item.spectrum_id),
    [selectedLibrarySpectra]
  )
  const selectedIdsKey = React.useMemo(() => selectedIds.join(','), [selectedIds])
  const selectedIdSet = React.useMemo(() => new Set(selectedIds), [selectedIds])

  const { data: searchData, isLoading, isFetching, error } = useLibrarySearchQuery(debouncedQuery, SEARCH_LIMIT)
  const {
    data: curveData,
    isFetching: curvesFetching,
    error: curvesError
  } = useLibraryCurvesQuery(imageId, selectedIds, !!imageId && selectedIds.length > 0)

  React.useEffect(() => {
    if (imageId && selectedIds.length > 0) {
      setSelectedLibraryCurves([])
    }
  }, [imageId, selectedIdsKey, selectedIds.length, setSelectedLibraryCurves])

  React.useEffect(() => {
    if (!imageId || selectedIds.length === 0) {
      setSelectedLibraryCurves([])
      return
    }
    if (curveData?.image_id === imageId) {
      setSelectedLibraryCurves(curveData.items)
    }
  }, [curveData, imageId, selectedIdsKey, selectedIds.length, setSelectedLibraryCurves])

  React.useEffect(() => {
    if (curvesError) {
      setSelectedLibraryCurves([])
    }
  }, [curvesError, setSelectedLibraryCurves])

  const results = searchData?.items ?? []
  const visibleSelectedCount = React.useMemo(
    () => results.filter((item) => selectedIdSet.has(item.spectrum_id)).length,
    [results, selectedIdSet]
  )
  const allVisibleSelected = results.length > 0 && visibleSelectedCount === results.length

  const renderSelectionHint = () => {
    if (selectedLibrarySpectra.length === 0) {
      return 'Search the library, tick reference spectra or use Select Visible, then click the image to match against the selected set.'
    }
    if (!imageId) {
      return `Selected ${selectedLibrarySpectra.length} spectra. Load an image to resample and draw them on the chart.`
    }
    return `Selected ${selectedLibrarySpectra.length} spectra. Clicking a pixel now matches only against this selected subset.`
  }

  const renderMeta = (item: LibrarySearchItem) => {
    const parts = [item.class ?? 'Unknown', item.sensor_type ?? null, item.measure_mode ?? null].filter(Boolean)
    if (parts.length === 0 && !item.source) {
      return 'No metadata'
    }
    return parts.concat(item.source ? [item.source] : []).join(' | ')
  }

  return (
    <div className="library-search-container panel">
      <div className="panel-header">
        <h3>Library Search</h3>
        <div className="library-search-header-side">
          <span className="library-search-counter">
            {selectedLibrarySpectra.length} selected
          </span>
          <button
            type="button"
            className="btn-icon"
            onClick={clearSelectedLibrarySpectra}
            disabled={selectedLibrarySpectra.length === 0}
            title="Clear selected spectra"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="library-search-toolbar">
        <label className="library-search-input-wrap">
          <Search size={15} />
          <input
            type="text"
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            placeholder="Search by name, class, source..."
          />
        </label>
        <div className="library-search-meta">
          <span>{searchData?.total ?? 0} found</span>
          <span>{(isLoading || isFetching) ? 'Searching...' : 'Ready'}</span>
        </div>
      </div>

      <div className="library-search-bulk-actions">
        <button
          type="button"
          className="btn btn-secondary library-bulk-btn"
          onClick={() => addSelectedLibrarySpectra(results)}
          disabled={results.length === 0 || allVisibleSelected}
        >
          Select Visible
        </button>
        <button
          type="button"
          className="btn-icon library-bulk-clear-btn"
          onClick={() => removeSelectedLibrarySpectra(results.map((item) => item.spectrum_id))}
          disabled={visibleSelectedCount === 0}
          title="Clear visible selections"
        >
          <X size={14} />
        </button>
      </div>

      <div className="library-search-note">
        {curvesFetching ? 'Resampling selected spectra for the current image...' : renderSelectionHint()}
      </div>

      {selectedLibrarySpectra.length > 0 && (
        <div className="library-selected-strip">
          {selectedLibrarySpectra.map((item) => (
            <button
              key={item.spectrum_id}
              type="button"
              className="library-selected-chip"
              onClick={() => toggleSelectedLibrarySpectrum(item)}
              title={item.name}
            >
              <span>{item.name}</span>
              <X size={12} />
            </button>
          ))}
        </div>
      )}

      <div className="library-search-results">
        {error ? (
          <div className="library-search-empty">Failed to search the spectral library.</div>
        ) : results.length === 0 && !isLoading ? (
          <div className="library-search-empty">No spectra matched the current query.</div>
        ) : (
          results.map((item) => {
            const checked = selectedIdSet.has(item.spectrum_id)
            return (
              <label
                key={item.spectrum_id}
                className={`library-search-row ${checked ? 'is-selected' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleSelectedLibrarySpectrum(item)}
                />
                <div className="library-search-row-body">
                  <div className="library-search-row-top">
                    <span className="library-search-name">{item.name}</span>
                    <span className="library-search-id">#{item.spectrum_id}</span>
                  </div>
                  <div className="library-search-row-meta">{renderMeta(item)}</div>
                </div>
              </label>
            )
          })
        )}
      </div>
    </div>
  )
}
