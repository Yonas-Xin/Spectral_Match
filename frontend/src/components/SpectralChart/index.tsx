import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { useAppStore } from '../../store/useAppStore'
import { useMatchPixelMutation } from '../../api/queries'
import { MATCH_CURVE_COLORS, TARGET_CURVE_COLOR } from '../../constants/spectralStyle'
import { Brush, Eraser } from 'lucide-react'
import './SpectralChart.css'

const MAX_TOOLTIP_CURVES = 6
const WATER_VAPOR_BANDS: Array<[number, number]> = [
  [1340, 1455],
  [1790, 1960],
  [2480, 2500]
]

export const SpectralChart: React.FC = () => {
  const chartRef = React.useRef<ReactECharts | null>(null)
  const [isMaskDrawMode, setIsMaskDrawMode] = React.useState(false)
  const [dragMaskRange, setDragMaskRange] = React.useState<{ start: number; end: number } | null>(null)
  const dragPointerIdRef = React.useRef<number | null>(null)
  const dragMaskRangeRef = React.useRef<{ start: number; end: number } | null>(null)
  const {
    imageId,
    selectedPixel,
    selection,
    selectionRevision,
    matchOptions,
    status,
    setMatchData,
    matchData,
    setStatus
  } = useAppStore()
  const { mutate: matchPixel, isPending } = useMatchPixelMutation()
  const { topN, metric, minValidBands, ignoreWaterBands, showWaterBandRanges, customMaskedRanges } = matchOptions
  const { setMatchOptions } = useAppStore()
  const waveBounds = React.useMemo(() => {
    const waves = matchData?.query?.wavelengths
    if (!Array.isArray(waves) || waves.length < 2) return null
    const min = Number(waves[0])
    const max = Number(waves[waves.length - 1])
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null
    return { min, max }
  }, [matchData])

  const mergeRanges = React.useCallback((ranges: Array<{ start: number; end: number }>) => {
    if (ranges.length === 0) return []
    const sorted = ranges
      .map((r) => ({ start: Math.min(r.start, r.end), end: Math.max(r.start, r.end) }))
      .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start)
      .sort((a, b) => a.start - b.start)
    if (sorted.length === 0) return []
    const merged: Array<{ start: number; end: number }> = [sorted[0]]
    for (let i = 1; i < sorted.length; i += 1) {
      const cur = sorted[i]
      const last = merged[merged.length - 1]
      if (cur.start <= last.end + 1e-6) {
        last.end = Math.max(last.end, cur.end)
      } else {
        merged.push({ ...cur })
      }
    }
    return merged
  }, [])

  const subtractRanges = React.useCallback(
    (
      source: Array<{ start: number; end: number }>,
      cut: Array<{ start: number; end: number }>
    ) => {
      if (source.length === 0) return []
      if (cut.length === 0) return source
      const out: Array<{ start: number; end: number }> = []
      for (const s of source) {
        let segments: Array<{ start: number; end: number }> = [{ ...s }]
        for (const c of cut) {
          const next: Array<{ start: number; end: number }> = []
          for (const seg of segments) {
            if (c.end <= seg.start || c.start >= seg.end) {
              next.push(seg)
              continue
            }
            if (c.start > seg.start) next.push({ start: seg.start, end: Math.min(c.start, seg.end) })
            if (c.end < seg.end) next.push({ start: Math.max(c.end, seg.start), end: seg.end })
          }
          segments = next
          if (segments.length === 0) break
        }
        out.push(...segments)
      }
      return mergeRanges(out)
    },
    [mergeRanges]
  )

  // Auto-trigger match when pixel changes
  React.useEffect(() => {
    if (!imageId || !selectedPixel) return
    if (status === 'loading_image') return

      matchPixel({
        image_id: imageId,
        x: selectedPixel.x,
        y: selectedPixel.y,
        top_n: topN,
        metric,
        ignore_water_bands: ignoreWaterBands,
        min_valid_bands: minValidBands,
        return_candidate_curves: true,
        selection: selection ?? { mode: 'pixel', x: selectedPixel.x, y: selectedPixel.y },
        custom_masked_ranges: customMaskedRanges
      }, {
        onSuccess: (data: any) => {
          setMatchData(data)
          setStatus('ready')
        },
        onError: (err: any) => {
          const message = String(err?.response?.data?.message || err?.message || '')
          if (message.includes('signature cache is building')) {
            setStatus('building_cache')
            return
          }
          setStatus('error', message || 'Match failed')
        }
      })
  }, [imageId, selectedPixel, selection, selectionRevision, topN, metric, minValidBands, ignoreWaterBands, customMaskedRanges, status, setMatchData, setStatus])

  React.useEffect(() => {
    if (!isMaskDrawMode) {
      dragPointerIdRef.current = null
      dragMaskRangeRef.current = null
      setDragMaskRange(null)
    }
  }, [isMaskDrawMode])

  const chartOptions = useMemo(() => {
    if (!matchData) {
      return {
        title: {
          text: 'No Data',
          left: 'center',
          textStyle: { color: '#94a3b8' }
        }
      }
    }

    const { query, results } = matchData
    const waves = query.wavelengths
    const waveMin = Number(waves[0])
    const waveMax = Number(waves[waves.length - 1])
    const waterBandAreas = WATER_VAPOR_BANDS
      .map(([start, end]) => {
        const left = Math.max(start, waveMin)
        const right = Math.min(end, waveMax)
        if (!Number.isFinite(left) || !Number.isFinite(right) || right <= left) return null
        return [{ xAxis: left }, { xAxis: right }]
      })
      .filter(Boolean) as Array<Array<{ xAxis: number }>>
    const dynamicRanges = dragMaskRange ? [...customMaskedRanges, dragMaskRange] : customMaskedRanges
    const customRangesMerged = mergeRanges(dynamicRanges)
    const customDisplayRanges = showWaterBandRanges
      ? subtractRanges(
          customRangesMerged,
          WATER_VAPOR_BANDS.map(([start, end]) => ({ start, end }))
        )
      : customRangesMerged
    const customBandAreas = customDisplayRanges
      .map(({ start, end }) => {
        const left = Math.max(start, waveMin)
        const right = Math.min(end, waveMax)
        if (!Number.isFinite(left) || !Number.isFinite(right) || right <= left) return null
        return [{ xAxis: left }, { xAxis: right }]
      })
      .filter(Boolean) as Array<Array<{ xAxis: number }>>
    const effectiveMaskedRanges = mergeRanges([
      ...(ignoreWaterBands ? WATER_VAPOR_BANDS.map(([start, end]) => ({ start, end })) : []),
      ...customMaskedRanges
    ])

    const pairSeries = (values: number[]) =>
      waves.map((w: number, i: number) => {
        const v = values[i]
        const masked = effectiveMaskedRanges.some(({ start, end }) => Number(w) >= start && Number(w) <= end)
        if (masked) return [w, null]
        return [w, v ?? null]
      })

    // Base target spectrum
    const seriesData: any[] = [
      {
        name: 'Target Pixel',
        type: 'line',
        data: pairSeries(query.spectrum),
        lineStyle: { width: 3, color: TARGET_CURVE_COLOR },
        symbol: 'none',
        showSymbol: false,
        z: 10,
        markArea: showWaterBandRanges
          ? {
              silent: true,
              itemStyle: {
                color: 'rgba(14, 116, 144, 0.12)'
              },
              label: {
                show: false
              },
              data: waterBandAreas
            }
          : undefined
      }
    ]

    // Matched spectra
    results.forEach((res: any, idx: number) => {
      if (res.curve) {
        seriesData.push({
          name: `${res.rank}. ${res.name}`,
          type: 'line',
          data: pairSeries(res.curve),
          lineStyle: { width: 1.5, type: 'dashed' as const },
          itemStyle: { color: MATCH_CURVE_COLORS[idx % MATCH_CURVE_COLORS.length] },
          symbol: 'none',
          showSymbol: false,
          z: 5 - idx
        })
      }
    })

    if (customBandAreas.length > 0) {
      seriesData.unshift({
        name: '__custom_mask__',
        type: 'line',
        data: [],
        silent: true,
        showSymbol: false,
        lineStyle: { opacity: 0 },
        tooltip: { show: false },
        markArea: {
          silent: true,
          itemStyle: {
            color: 'rgba(239, 68, 68, 0.14)'
          },
          label: { show: false },
          data: customBandAreas
        }
      })
    }

    const allYValues = seriesData.flatMap((series) =>
      (series.data as Array<[number, number | null]>)
        .map((p) => p[1])
        .filter((v) => typeof v === 'number' && Number.isFinite(v)) as number[]
    )
    let yMin = 0
    let yMax = 1
    if (allYValues.length > 0) {
      const rawMin = Math.min(...allYValues)
      const rawMax = Math.max(...allYValues)
      if (rawMax > rawMin) {
        const pad = (rawMax - rawMin) * 0.06
        yMin = rawMin - pad
        yMax = rawMax + pad
      } else {
        const pad = Math.max(Math.abs(rawMin) * 0.08, 0.05)
        yMin = rawMin - pad
        yMax = rawMax + pad
      }
    }
    let yMinAxis = Math.floor(yMin * 10) / 10
    let yMaxAxis = Math.ceil(yMax * 10) / 10
    yMinAxis = Math.max(0, yMinAxis)
    if (yMaxAxis <= yMinAxis) {
      yMaxAxis = yMinAxis + 0.1
    }
    const xLabelFormatter = (value: number) => {
      const n = Number(value)
      if (!Number.isFinite(n)) return ''
      const nearest500 = Math.round(n / 500) * 500
      if (Math.abs(n - nearest500) > 1) return ''
      if (nearest500 < waveMin - 1 || nearest500 > waveMax + 1) return ''
      return `${nearest500}`
    }

    const getSeriesRank = (seriesName: string): number => {
      if (seriesName === 'Target Pixel') return -1
      const m = /^(\d+)\./.exec(seriesName)
      return m ? Number(m[1]) : Number.MAX_SAFE_INTEGER
    }

    const tooltipFormatter = (params: any) => {
      const items = Array.isArray(params) ? params : [params]
      if (items.length === 0) return ''

      const wavelength = Number(items[0]?.axisValue)
      const target = items.find((it: any) => it.seriesName === 'Target Pixel')
      const others = items
        .filter((it: any) => it.seriesName !== 'Target Pixel')
        .sort((a: any, b: any) => getSeriesRank(String(a.seriesName)) - getSeriesRank(String(b.seriesName)))
      const ordered = [target, ...others].filter(Boolean)
      const visible = ordered.slice(0, MAX_TOOLTIP_CURVES)
      const hiddenCount = Math.max(0, ordered.length - visible.length)

      const rows = visible
        .map((it: any) => {
          const point = Array.isArray(it.value) ? it.value[1] : it.value
          const numeric = typeof point === 'number' ? point : Number(point)
          const valueText = Number.isFinite(numeric) ? numeric.toFixed(4) : '-'
          return (
            `<div class="spectral-tooltip-row">` +
              `<span class="spectral-tooltip-name">${it.marker}${it.seriesName}</span>` +
              `<span class="spectral-tooltip-value">${valueText}</span>` +
            `</div>`
          )
        })
        .join('')

      const title = Number.isFinite(wavelength)
        ? `Wavelength: ${wavelength.toFixed(2)} nm`
        : 'Wavelength'
      const more = hiddenCount > 0
        ? `<div class="spectral-tooltip-more">+${hiddenCount} more curves...</div>`
        : ''

      return `<div class="spectral-tooltip-inner"><div class="spectral-tooltip-title">${title}</div>${rows}${more}</div>`
    }

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        appendToBody: true,
        className: 'spectral-tooltip-pop',
        backgroundColor: 'rgba(255, 255, 255, 0.96)',
        borderColor: 'rgba(15, 42, 61, 0.2)',
        borderWidth: 1,
        textStyle: { color: '#1f3344', fontSize: 12 },
        extraCssText: 'border-radius: 10px; box-shadow: 0 10px 24px rgba(15,42,61,0.18);',
        formatter: tooltipFormatter
      },
      grid: {
        left: 64,
        right: 64,
        top: 42,
        bottom: 62,
        containLabel: true
      },
      xAxis: {
        type: 'value',
        name: 'Wavelength (nm)',
        nameLocation: 'middle',
        nameGap: 34,
        min: waves[0],
        max: waves[waves.length - 1],
        axisPointer: {
          label: {
            formatter: ({ value }: any) => {
              const n = Number(value)
              return Number.isFinite(n) ? n.toFixed(2) : ''
            }
          }
        },
        axisLine: { lineStyle: { color: '#6a879d' } },
        axisLabel: {
          color: '#547085',
          formatter: xLabelFormatter
        },
        splitLine: { lineStyle: { color: 'rgba(15,42,61,0.08)' } }
      },
      yAxis: {
        type: 'value',
        name: 'Reflectance',
        min: yMinAxis,
        max: yMaxAxis,
        axisPointer: {
          label: {
            formatter: ({ value }: any) => {
              const n = Number(value)
              return Number.isFinite(n) ? n.toFixed(4) : ''
            }
          }
        },
        axisLine: { lineStyle: { color: '#6a879d' } },
        axisLabel: {
          color: '#547085',
          formatter: (value: number) => Number(value).toFixed(1)
        },
        splitLine: { lineStyle: { color: 'rgba(15,42,61,0.08)' } }
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: [0] },
        {
          type: 'slider',
          xAxisIndex: [0],
          bottom: 14,
          height: 22,
          borderColor: 'rgba(15,42,61,0.12)',
          fillerColor: 'rgba(13,148,136,0.18)',
          textStyle: { color: '#547085' }
        }
      ],
      series: seriesData,
      backgroundColor: 'transparent'
    }
  }, [matchData, ignoreWaterBands, showWaterBandRanges, customMaskedRanges, dragMaskRange, mergeRanges, subtractRanges])

  const clientToWave = React.useCallback((clientX: number, _clientY: number): number | null => {
    const inst = chartRef.current?.getEchartsInstance()
    if (!inst || !waveBounds) return null
    const rect = (inst.getDom() as HTMLElement).getBoundingClientRect()
    const px = clientX - rect.left
    if (!Number.isFinite(px)) return null
    const left = Number(inst.convertToPixel({ xAxisIndex: 0 }, waveBounds.min))
    const right = Number(inst.convertToPixel({ xAxisIndex: 0 }, waveBounds.max))
    if (!Number.isFinite(left) || !Number.isFinite(right) || Math.abs(right - left) < 1e-6) return null
    const x0 = Math.min(left, right)
    const x1 = Math.max(left, right)
    const clipped = Math.max(x0, Math.min(x1, px))
    const t = (clipped - x0) / (x1 - x0)
    const w = waveBounds.min + t * (waveBounds.max - waveBounds.min)
    if (!Number.isFinite(w)) return null
    return w
  }, [waveBounds])

  const finalizeOverlayDrag = React.useCallback(() => {
    const range = dragMaskRangeRef.current
    if (!range) return
    const start = Math.min(range.start, range.end)
    const end = Math.max(range.start, range.end)
    if (end - start >= 2) {
      const currentRanges = useAppStore.getState().matchOptions.customMaskedRanges
      setMatchOptions({
        customMaskedRanges: mergeRanges([...currentRanges, { start, end }])
      })
    }
    dragMaskRangeRef.current = null
    setDragMaskRange(null)
  }, [setMatchOptions, mergeRanges])

  const onOverlayPointerDown = React.useCallback((evt: React.PointerEvent<HTMLDivElement>) => {
    if (!isMaskDrawMode) return
    if (evt.button === 2) return
    if (dragPointerIdRef.current != null) return
    const wave = clientToWave(evt.clientX, evt.clientY)
    if (wave == null) return
    dragPointerIdRef.current = evt.pointerId
    try {
      evt.currentTarget.setPointerCapture(evt.pointerId)
    } catch {
    }
    const next = { start: wave, end: wave }
    dragMaskRangeRef.current = next
    setDragMaskRange(next)
    evt.preventDefault()
    evt.stopPropagation()
  }, [isMaskDrawMode, clientToWave])

  const onOverlayPointerMove = React.useCallback((evt: React.PointerEvent<HTMLDivElement>) => {
    if (dragPointerIdRef.current == null) return
    if (dragPointerIdRef.current === -1) return
    const wave = clientToWave(evt.clientX, evt.clientY)
    if (wave == null) return
    setDragMaskRange((prev) => {
      const base = prev ?? dragMaskRangeRef.current
      if (!base) return prev
      const next = { start: base.start, end: wave }
      dragMaskRangeRef.current = next
      return next
    })
    evt.preventDefault()
  }, [clientToWave])

  const onOverlayPointerUp = React.useCallback((evt: React.PointerEvent<HTMLDivElement>) => {
    if (dragPointerIdRef.current == null) return
    if (dragPointerIdRef.current === -1) return
    try {
      evt.currentTarget.releasePointerCapture(evt.pointerId)
    } catch {
    }
    dragPointerIdRef.current = null
    finalizeOverlayDrag()
    evt.preventDefault()
    evt.stopPropagation()
  }, [finalizeOverlayDrag])

  React.useEffect(() => {
    if (!isMaskDrawMode) return
    const onWindowPointerUp = () => {
      if (dragPointerIdRef.current == null) return
      dragPointerIdRef.current = null
      finalizeOverlayDrag()
    }
    window.addEventListener('pointerup', onWindowPointerUp)
    return () => {
      window.removeEventListener('pointerup', onWindowPointerUp)
    }
  }, [isMaskDrawMode, finalizeOverlayDrag])

  const onOverlayMouseDown = React.useCallback((evt: React.MouseEvent<HTMLDivElement>) => {
    if (!isMaskDrawMode) return
    if (evt.button === 2) return
    if (dragPointerIdRef.current != null) return
    const wave = clientToWave(evt.clientX, evt.clientY)
    if (wave == null) return
    dragPointerIdRef.current = -1
    const next = { start: wave, end: wave }
    dragMaskRangeRef.current = next
    setDragMaskRange(next)
    evt.preventDefault()
    evt.stopPropagation()
  }, [isMaskDrawMode, clientToWave])

  const onOverlayMouseMove = React.useCallback((evt: React.MouseEvent<HTMLDivElement>) => {
    if (dragPointerIdRef.current !== -1) return
    const wave = clientToWave(evt.clientX, evt.clientY)
    if (wave == null) return
    setDragMaskRange((prev) => {
      const base = prev ?? dragMaskRangeRef.current
      if (!base) return prev
      const next = { start: base.start, end: wave }
      dragMaskRangeRef.current = next
      return next
    })
    evt.preventDefault()
  }, [clientToWave])

  const onOverlayMouseUp = React.useCallback((evt: React.MouseEvent<HTMLDivElement>) => {
    if (dragPointerIdRef.current !== -1) return
    dragPointerIdRef.current = null
    finalizeOverlayDrag()
    evt.preventDefault()
    evt.stopPropagation()
  }, [finalizeOverlayDrag])

  React.useEffect(() => {
    if (!isMaskDrawMode) return
    const onWindowMouseUp = () => {
      if (dragPointerIdRef.current !== -1) return
      dragPointerIdRef.current = null
      finalizeOverlayDrag()
    }
    window.addEventListener('mouseup', onWindowMouseUp)
    return () => {
      window.removeEventListener('mouseup', onWindowMouseUp)
    }
  }, [isMaskDrawMode, finalizeOverlayDrag])

  const legendItems = useMemo(() => {
    if (!matchData?.results) return []
    return [
      { key: 'target', color: TARGET_CURVE_COLOR, label: 'Target Pixel', dashed: false },
      ...matchData.results
        .filter((res: any) => Array.isArray(res.curve) && res.curve.length > 0)
        .map((res: any, idx: number) => ({
          key: `match-${res.rank}`,
          color: MATCH_CURVE_COLORS[idx % MATCH_CURVE_COLORS.length],
          label: `${res.rank}. ${res.name}`,
          dashed: true
        }))
    ]
  }, [matchData])

  return (
    <div className="spectral-chart-container panel">
      <div className="panel-header">
        <h3>Spectral Comparison</h3>
        <div className="chart-actions">
          <button
            type="button"
            className={`btn-icon chart-action-btn ${isMaskDrawMode ? 'active' : ''}`}
            onClick={() => setIsMaskDrawMode((v) => !v)}
            title="Draw masked ranges"
          >
            <Brush size={14} />
          </button>
          <button
            type="button"
            className="btn-icon chart-action-btn"
            onClick={() => {
              dragMaskRangeRef.current = null
              setMatchOptions({ customMaskedRanges: [] })
            }}
            title="Clear masked ranges"
            disabled={customMaskedRanges.length === 0}
          >
            <Eraser size={14} />
          </button>
          {isPending && <span className="text-accent text-sm ml-2">Matching...</span>}
        </div>
      </div>
      <div className="chart-wrapper">
        <div className="chart-inner">
          <ReactECharts
            ref={chartRef}
            option={chartOptions}
            style={{ height: '100%', width: '100%' }}
            notMerge={true}
            lazyUpdate={true}
          />
          {isMaskDrawMode && (
            <div
              className="chart-mask-overlay"
              onPointerDown={onOverlayPointerDown}
              onPointerMove={onOverlayPointerMove}
              onPointerUp={onOverlayPointerUp}
              onPointerCancel={onOverlayPointerUp}
              onMouseDown={onOverlayMouseDown}
              onMouseMove={onOverlayMouseMove}
              onMouseUp={onOverlayMouseUp}
              onContextMenu={(e) => e.preventDefault()}
            />
          )}
        </div>
        <div className="chart-side-legend" aria-label="Curve legend">
          {legendItems.length === 0 ? (
            <div className="chart-legend-empty">Select a pixel to show legend.</div>
          ) : (
            legendItems.map((item: any) => (
              <div className="chart-legend-item" key={item.key} title={item.label}>
                <span
                  className={`chart-legend-line ${item.dashed ? 'is-dashed' : ''}`}
                  style={{ ['--legend-color' as any]: item.color }}
                />
                <span className="chart-legend-label">{item.label}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
