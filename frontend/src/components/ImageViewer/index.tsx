import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAppStore, type SelectionMode } from '../../store/useAppStore'
import {
  ZoomIn,
  ZoomOut,
  Maximize,
  MousePointer2,
  Square,
  Circle,
  PenLine
} from 'lucide-react'
import './ImageViewer.css'

type Point = { x: number; y: number }
type BoxDraft = { start: Point; end: Point }
type CircleDraft = { center: Point; radius: number }

const pointDistance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y)

const pointInPolygon = (x: number, y: number, points: Point[]) => {
  let inside = false
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i].x
    const yi = points[i].y
    const xj = points[j].x
    const yj = points[j].y
    const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

const estimateCirclePixelCount = (
  cx: number,
  cy: number,
  radius: number,
  samples: number,
  lines: number
) => {
  const x0 = Math.max(0, Math.floor(cx - radius))
  const x1 = Math.min(samples - 1, Math.ceil(cx + radius))
  const y0 = Math.max(0, Math.floor(cy - radius))
  const y1 = Math.min(lines - 1, Math.ceil(cy + radius))
  let count = 0
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) count += 1
    }
  }
  return Math.max(1, count)
}

const estimateLassoPixelCount = (points: Point[], samples: number, lines: number) => {
  if (points.length < 3) return 1
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const x0 = Math.max(0, Math.floor(Math.min(...xs)))
  const x1 = Math.min(samples - 1, Math.ceil(Math.max(...xs)))
  const y0 = Math.max(0, Math.floor(Math.min(...ys)))
  const y1 = Math.min(lines - 1, Math.ceil(Math.max(...ys)))
  let count = 0
  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      if (pointInPolygon(x + 0.5, y + 0.5, points)) count += 1
    }
  }
  return Math.max(1, count)
}

export const ImageViewer: React.FC = () => {
  const {
    imageId,
    imageMeta,
    selectedPixel,
    selection,
    selectionMode,
    setSelectedPixel,
    setSelectionMode,
    setSelectionRegion
  } = useAppStore()
  const stageRef = useRef<HTMLDivElement>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const toolbarRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)

  const [scale, setScale] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [isSelecting, setIsSelecting] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 })
  const [wrapperFrame, setWrapperFrame] = useState({ width: 640, height: 640 })
  const [imageRevision, setImageRevision] = useState(0)
  const [boxDraft, setBoxDraft] = useState<BoxDraft | null>(null)
  const [circleDraft, setCircleDraft] = useState<CircleDraft | null>(null)
  const [lassoDraft, setLassoDraft] = useState<Point[] | null>(null)

  const previewUrl = imageId ? `http://127.0.0.1:8000/api/v1/image/preview/${imageId}.png` : null
  const imageAspect = useMemo(() => {
    const w = Number(imageMeta?.samples || 0)
    const h = Number(imageMeta?.lines || 0)
    if (w > 0 && h > 0) {
      return w / h
    }
    return 1
  }, [imageMeta?.samples, imageMeta?.lines])

  const modeButtons = useMemo(
    () => [
      { mode: 'pixel' as SelectionMode, label: 'Pixel', icon: MousePointer2 },
      { mode: 'box' as SelectionMode, label: 'Box', icon: Square },
      { mode: 'circle' as SelectionMode, label: 'Circle', icon: Circle },
      { mode: 'lasso' as SelectionMode, label: 'Lasso', icon: PenLine }
    ],
    []
  )

  const getCanvasPoint = (
    e: React.MouseEvent<HTMLCanvasElement> | React.WheelEvent<HTMLCanvasElement>
  ) => {
    const canvas = canvasRef.current
    const rect = canvas?.getBoundingClientRect()
    if (!canvas || !rect || rect.width <= 0 || rect.height <= 0) return null
    return {
      x: ((e.clientX - rect.left) * canvas.width) / rect.width,
      y: ((e.clientY - rect.top) * canvas.height) / rect.height
    }
  }

  const clampImagePoint = useCallback(
    (pt: Point): Point => {
      if (!imageMeta) return pt
      return {
        x: Math.min(Math.max(pt.x, 0), imageMeta.samples - 1),
        y: Math.min(Math.max(pt.y, 0), imageMeta.lines - 1)
      }
    },
    [imageMeta]
  )

  const canvasPointToImage = useCallback(
    (pt: Point): Point => ({ x: (pt.x - offset.x) / scale, y: (pt.y - offset.y) / scale }),
    [offset, scale]
  )

  const imagePointToCanvas = useCallback(
    (pt: Point): Point => ({ x: offset.x + pt.x * scale, y: offset.y + pt.y * scale }),
    [offset, scale]
  )

  const fitImageToViewport = useCallback(
    (img: HTMLImageElement) => {
      const viewW = canvasSize.width
      const viewH = canvasSize.height
      const sourceW = Number(imageMeta?.samples || img.width)
      const sourceH = Number(imageMeta?.lines || img.height)
      if (viewW <= 0 || viewH <= 0 || sourceW <= 0 || sourceH <= 0) {
        setScale(1)
        setOffset({ x: 0, y: 0 })
        return
      }
      const fitScale = Math.min(viewW / sourceW, viewH / sourceH)
      const nextScale = Number.isFinite(fitScale) && fitScale > 0 ? fitScale : 1
      setScale(nextScale)
      setOffset({
        x: (viewW - sourceW * nextScale) / 2,
        y: (viewH - sourceH * nextScale) / 2
      })
    },
    [canvasSize.height, canvasSize.width, imageMeta?.lines, imageMeta?.samples]
  )

  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return

    const toNum = (v: string | null | undefined) => {
      const n = Number.parseFloat(v || '0')
      return Number.isFinite(n) ? n : 0
    }

    const updateFrame = () => {
      const toolbar = toolbarRef.current
      const cs = window.getComputedStyle(stage)
      const isColumn = cs.flexDirection.startsWith('column')
      const padX = toNum(cs.paddingLeft) + toNum(cs.paddingRight)
      const padY = toNum(cs.paddingTop) + toNum(cs.paddingBottom)
      const gap = isColumn ? toNum(cs.rowGap || cs.gap) : toNum(cs.columnGap || cs.gap)

      const contentW = Math.max(1, stage.clientWidth - padX)
      const contentH = Math.max(1, stage.clientHeight - padY)
      const toolbarW = toolbar && previewUrl && !isColumn ? toolbar.getBoundingClientRect().width : 0
      const toolbarH = toolbar && previewUrl && isColumn ? toolbar.getBoundingClientRect().height : 0

      const availableW = Math.max(1, contentW - toolbarW - (toolbarW > 0 ? gap : 0))
      const availableH = Math.max(1, contentH - toolbarH - (toolbarH > 0 ? gap : 0))
      const ratio = imageAspect > 0 ? imageAspect : 1

      let fitW = availableW
      let fitH = fitW / ratio
      if (fitH > availableH) {
        fitH = availableH
        fitW = fitH * ratio
      }

      const width = Math.max(1, Math.floor(fitW))
      const height = Math.max(1, Math.floor(fitH))
      setWrapperFrame((prev) => (prev.width === width && prev.height === height ? prev : { width, height }))
    }

    updateFrame()
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(updateFrame)
      observer.observe(stage)
      if (toolbarRef.current) observer.observe(toolbarRef.current)
      return () => observer.disconnect()
    }
    window.addEventListener('resize', updateFrame)
    return () => window.removeEventListener('resize', updateFrame)
  }, [previewUrl, imageAspect])

  useEffect(() => {
    const wrapper = wrapperRef.current
    const canvas = canvasRef.current
    if (!wrapper || !canvas) return

    const updateCanvasSize = () => {
      const rect = wrapper.getBoundingClientRect()
      const width = Math.max(1, Math.floor(rect.width))
      const height = Math.max(1, Math.floor(rect.height))
      if (canvas.width !== width) canvas.width = width
      if (canvas.height !== height) canvas.height = height
      setCanvasSize((prev) => (prev.width === width && prev.height === height ? prev : { width, height }))
    }

    updateCanvasSize()

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(updateCanvasSize)
      observer.observe(wrapper)
      return () => observer.disconnect()
    }

    window.addEventListener('resize', updateCanvasSize)
    return () => window.removeEventListener('resize', updateCanvasSize)
  }, [wrapperFrame.width, wrapperFrame.height])

  useEffect(() => {
    if (!previewUrl) {
      imageRef.current = null
      return
    }
    const img = new Image()
    img.crossOrigin = 'Anonymous'
    img.src = previewUrl
    img.onload = () => {
      imageRef.current = img
      fitImageToViewport(img)
      setImageRevision((v) => v + 1)
    }
  }, [previewUrl, fitImageToViewport])

  useEffect(() => {
    const canvas = canvasRef.current
    const img = imageRef.current
    if (!canvas || !img) return

    const drawBox = (ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number) => {
      const p0 = imagePointToCanvas({ x: Math.min(x0, x1), y: Math.min(y0, y1) })
      const p1 = imagePointToCanvas({ x: Math.max(x0, x1) + 1, y: Math.max(y0, y1) + 1 })
      ctx.save()
      ctx.strokeStyle = '#0f766e'
      ctx.fillStyle = 'rgba(13, 148, 136, 0.18)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4])
      ctx.fillRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y)
      ctx.strokeRect(p0.x, p0.y, p1.x - p0.x, p1.y - p0.y)
      ctx.restore()
    }

    const drawCircle = (ctx: CanvasRenderingContext2D, cx: number, cy: number, radius: number) => {
      const center = imagePointToCanvas({ x: cx + 0.5, y: cy + 0.5 })
      ctx.save()
      ctx.strokeStyle = '#0f766e'
      ctx.fillStyle = 'rgba(13, 148, 136, 0.18)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      ctx.arc(center.x, center.y, Math.max(1, radius * scale), 0, 2 * Math.PI)
      ctx.fill()
      ctx.stroke()
      ctx.restore()
    }

    const drawLasso = (ctx: CanvasRenderingContext2D, points: Point[]) => {
      if (points.length < 2) return
      ctx.save()
      ctx.strokeStyle = '#0f766e'
      ctx.fillStyle = 'rgba(13, 148, 136, 0.18)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 4])
      const start = imagePointToCanvas(points[0])
      ctx.beginPath()
      ctx.moveTo(start.x, start.y)
      for (let i = 1; i < points.length; i += 1) {
        const p = imagePointToCanvas(points[i])
        ctx.lineTo(p.x, p.y)
      }
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      ctx.restore()
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.imageSmoothingEnabled = false
    const sourceW = Number(imageMeta?.samples || img.width)
    const sourceH = Number(imageMeta?.lines || img.height)
    ctx.drawImage(img, 0, 0, img.width, img.height, offset.x, offset.y, sourceW * scale, sourceH * scale)

    if (selection?.mode === 'box') {
      drawBox(ctx, selection.x0, selection.y0, selection.x1, selection.y1)
    } else if (selection?.mode === 'circle') {
      drawCircle(ctx, selection.cx, selection.cy, selection.radius)
    } else if (selection?.mode === 'lasso') {
      drawLasso(ctx, selection.points)
    }

    if (boxDraft) {
      drawBox(ctx, boxDraft.start.x, boxDraft.start.y, boxDraft.end.x, boxDraft.end.y)
    }
    if (circleDraft) {
      drawCircle(ctx, circleDraft.center.x, circleDraft.center.y, circleDraft.radius)
    }
    if (lassoDraft && lassoDraft.length >= 2) {
      drawLasso(ctx, lassoDraft)
    }

    if (selectedPixel) {
      const px = offset.x + (selectedPixel.x + 0.5) * scale
      const py = offset.y + (selectedPixel.y + 0.5) * scale

      ctx.save()
      ctx.strokeStyle = '#ef4444'
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(px - 10, py)
      ctx.lineTo(px + 10, py)
      ctx.moveTo(px, py - 10)
      ctx.lineTo(px, py + 10)
      ctx.stroke()
      ctx.beginPath()
      ctx.arc(px, py, 4.2, 0, 2 * Math.PI)
      ctx.stroke()
      ctx.restore()
    }
  }, [scale, offset, selectedPixel, selection, boxDraft, circleDraft, lassoDraft, imageRevision, canvasSize, imagePointToCanvas, imageMeta?.lines, imageMeta?.samples])

  useEffect(() => {
    setBoxDraft(null)
    setCircleDraft(null)
    setLassoDraft(null)
    setIsSelecting(false)
  }, [selectionMode, imageId])

  const finalizeRegionSelection = useCallback(() => {
    if (!imageMeta) return

    if (boxDraft) {
      const x0 = Math.min(Math.round(boxDraft.start.x), Math.round(boxDraft.end.x))
      const x1 = Math.max(Math.round(boxDraft.start.x), Math.round(boxDraft.end.x))
      const y0 = Math.min(Math.round(boxDraft.start.y), Math.round(boxDraft.end.y))
      const y1 = Math.max(Math.round(boxDraft.start.y), Math.round(boxDraft.end.y))
      const cx = Math.round((x0 + x1) / 2)
      const cy = Math.round((y0 + y1) / 2)
      const count = Math.max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
      setSelectionRegion({ mode: 'box', x0, y0, x1, y1 }, { x: cx, y: cy }, count)
      setBoxDraft(null)
      return
    }

    if (circleDraft) {
      const cx = Math.round(circleDraft.center.x)
      const cy = Math.round(circleDraft.center.y)
      const radius = Math.max(1, Number(circleDraft.radius.toFixed(2)))
      const count = estimateCirclePixelCount(cx, cy, radius, imageMeta.samples, imageMeta.lines)
      setSelectionRegion({ mode: 'circle', cx, cy, radius }, { x: cx, y: cy }, count)
      setCircleDraft(null)
      return
    }

    if (lassoDraft && lassoDraft.length >= 3) {
      const points = lassoDraft.map((p) => ({
        x: Number(Math.max(0, Math.min(imageMeta.samples - 1, p.x)).toFixed(2)),
        y: Number(Math.max(0, Math.min(imageMeta.lines - 1, p.y)).toFixed(2))
      }))
      const centerX = Math.round(points.reduce((acc, p) => acc + p.x, 0) / points.length)
      const centerY = Math.round(points.reduce((acc, p) => acc + p.y, 0) / points.length)
      const count = estimateLassoPixelCount(points, imageMeta.samples, imageMeta.lines)
      setSelectionRegion({ mode: 'lasso', points }, { x: centerX, y: centerY }, count)
    }
    setLassoDraft(null)
  }, [boxDraft, circleDraft, lassoDraft, imageMeta, setSelectionRegion])

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getCanvasPoint(e)
    if (!point) return

    if (e.button === 1 || e.button === 2 || e.shiftKey) {
      setIsDragging(true)
      setDragStart({ x: point.x - offset.x, y: point.y - offset.y })
      return
    }

    if (e.button !== 0 || !imageMeta) return
    const imgPoint = clampImagePoint(canvasPointToImage(point))

    if (selectionMode === 'pixel') {
      const rawX = Math.floor(imgPoint.x)
      const rawY = Math.floor(imgPoint.y)
      if (rawX >= 0 && rawX < imageMeta.samples && rawY >= 0 && rawY < imageMeta.lines) {
        setSelectedPixel(rawX, rawY)
      }
      return
    }

    setIsSelecting(true)
    if (selectionMode === 'box') {
      setBoxDraft({ start: imgPoint, end: imgPoint })
    } else if (selectionMode === 'circle') {
      setCircleDraft({ center: imgPoint, radius: 1 })
    } else if (selectionMode === 'lasso') {
      setLassoDraft([imgPoint])
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const point = getCanvasPoint(e)
    if (!point) return

    if (isDragging) {
      setOffset({
        x: point.x - dragStart.x,
        y: point.y - dragStart.y
      })
      return
    }

    if (!isSelecting || !imageMeta) return
    const imgPoint = clampImagePoint(canvasPointToImage(point))

    if (boxDraft) {
      setBoxDraft((prev) => (prev ? { ...prev, end: imgPoint } : prev))
      return
    }

    if (circleDraft) {
      setCircleDraft((prev) => {
        if (!prev) return prev
        return { ...prev, radius: Math.max(1, pointDistance(prev.center, imgPoint)) }
      })
      return
    }

    if (lassoDraft) {
      setLassoDraft((prev) => {
        if (!prev || prev.length === 0) return prev
        const last = prev[prev.length - 1]
        if (pointDistance(last, imgPoint) < 0.9) return prev
        return [...prev, imgPoint]
      })
    }
  }

  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false)
    }
    if (isSelecting) {
      finalizeRegionSelection()
      setIsSelecting(false)
    }
  }

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()

    const point = getCanvasPoint(e)
    if (!point) return

    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9
    const newScale = Math.max(0.1, Math.min(scale * zoomFactor, 20))

    const newOffsetX = point.x - (point.x - offset.x) * (newScale / scale)
    const newOffsetY = point.y - (point.y - offset.y) * (newScale / scale)

    setScale(newScale)
    setOffset({ x: newOffsetX, y: newOffsetY })
  }

  const resetView = () => {
    const img = imageRef.current
    if (img) {
      fitImageToViewport(img)
    } else {
      setScale(1)
      setOffset({ x: 0, y: 0 })
    }
  }

  return (
    <div className="image-viewer-container panel">
      <div className="panel-header">
        <h3>Image Viewer</h3>
        <div className="viewer-controls">
          <button className="btn-icon" onClick={() => setScale((s) => s * 1.1)}>
            <ZoomIn size={16} />
          </button>
          <button className="btn-icon" onClick={() => setScale((s) => Math.max(0.1, s * 0.9))}>
            <ZoomOut size={16} />
          </button>
          <button className="btn-icon" onClick={resetView}>
            <Maximize size={16} />
          </button>
        </div>
      </div>

      <div className="viewer-stage" ref={stageRef}>
        <div
          className="canvas-wrapper"
          ref={wrapperRef}
          style={{
            width: `${wrapperFrame.width}px`,
            height: `${wrapperFrame.height}px`
          }}
          onWheelCapture={(e) => {
            e.preventDefault()
          }}
        >
          {previewUrl ? (
            <canvas
              ref={canvasRef}
              width={800}
              height={600}
              className={`viewer-canvas mode-${selectionMode}`}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
              onContextMenu={(e) => e.preventDefault()}
            />
          ) : (
            <div className="empty-state">
              <p>No image loaded. Please load a hyperspectral image from the sidebar.</p>
            </div>
          )}
        </div>

        {previewUrl && (
          <div className="selection-toolbar" ref={toolbarRef} aria-label="Selection tools">
            {modeButtons.map((item) => {
              const Icon = item.icon
              return (
                <button
                  key={item.mode}
                  type="button"
                  className={`selection-btn ${selectionMode === item.mode ? 'active' : ''}`}
                  onClick={() => setSelectionMode(item.mode)}
                  title={item.label}
                >
                  <Icon size={15} />
                  <span>{item.label}</span>
                </button>
              )
            })}
          </div>
        )}
      </div>

      <div className="viewer-hint">
        Pixel/Box/Circle/Lasso: left drag or click | Shift/Middle/Right Drag: Pan | Scroll: Zoom
      </div>
    </div>
  )
}
