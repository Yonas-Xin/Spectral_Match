import { create } from 'zustand'
import type { LibraryCurveItem, LibrarySearchItem, MatchData } from '../types/spectral'

export type SelectionMode = 'pixel' | 'box' | 'circle' | 'lasso'

export type MatchSelection =
  | { mode: 'pixel'; x: number; y: number }
  | { mode: 'box'; x0: number; y0: number; x1: number; y1: number }
  | { mode: 'circle'; cx: number; cy: number; radius: number }
  | { mode: 'lasso'; points: Array<{ x: number; y: number }> }

export interface AppState {
  imageId: string | null;
  imagePath: string | null;
  imageMeta: any | null;
  selectedPixel: { x: number; y: number } | null;
  selectedPixelCount: number;
  selection: MatchSelection | null;
  selectionMode: SelectionMode;
  selectionRevision: number;
  signatureHash: string | null;
  status: 'idle' | 'loading_image' | 'building_cache' | 'ready' | 'error';
  errorMessage: string | null;
  matchData: MatchData | null;
  selectedLibrarySpectra: LibrarySearchItem[];
  selectedLibraryCurves: LibraryCurveItem[];
  matchOptions: {
    topN: number;
    metric: string;
    minValidBands: number;
    ignoreWaterBands: boolean;
    showWaterBandRanges: boolean;
    customMaskedRanges: Array<{ start: number; end: number }>;
  };

  setImageData: (data: { imageId: string; path: string; meta: any; signatureHash: string }) => void;
  setSelectedPixel: (x: number, y: number) => void;
  setSelectionRegion: (
    selection: Exclude<MatchSelection, { mode: 'pixel' }>,
    center: { x: number; y: number },
    pixelCount: number
  ) => void;
  setSelectionMode: (mode: SelectionMode) => void;
  clearSelection: () => void;
  setStatus: (status: AppState['status'], errorMessage?: string | null) => void;
  setMatchOptions: (options: Partial<AppState['matchOptions']>) => void;
  setMatchData: (data: MatchData | null) => void;
  toggleSelectedLibrarySpectrum: (item: LibrarySearchItem) => void;
  addSelectedLibrarySpectra: (items: LibrarySearchItem[]) => void;
  removeSelectedLibrarySpectra: (spectrumIds: number[]) => void;
  clearSelectedLibrarySpectra: () => void;
  setSelectedLibraryCurves: (items: LibraryCurveItem[]) => void;
}

export const useAppStore = create<AppState>((set) => ({
  imageId: null,
  imagePath: null,
  imageMeta: null,
  selectedPixel: null,
  selectedPixelCount: 0,
  selection: null,
  selectionMode: 'pixel',
  selectionRevision: 0,
  signatureHash: null,
  status: 'idle',
  errorMessage: null,
  matchData: null,
  selectedLibrarySpectra: [],
  selectedLibraryCurves: [],
  matchOptions: {
    topN: 10,
    metric: 'sam',
    minValidBands: 20,
    ignoreWaterBands: true,
    showWaterBandRanges: true,
    customMaskedRanges: []
  },

  setImageData: ({ imageId, path, meta, signatureHash }) => 
    set({
      imageId,
      imagePath: path,
      imageMeta: meta,
      signatureHash,
      selectedPixel: null,
      selectedPixelCount: 0,
      selection: null,
      selectionRevision: 0,
      status: 'ready',
      errorMessage: null,
      matchData: null,
      selectedLibraryCurves: []
    }),
    
  setSelectedPixel: (x, y) =>
    set((state) => ({
      selectedPixel: { x, y },
      selectedPixelCount: 1,
      selection: { mode: 'pixel', x, y },
      selectionRevision: state.selectionRevision + 1
    })),

  setSelectionRegion: (selection, center, pixelCount) =>
    set((state) => ({
      selectedPixel: center,
      selectedPixelCount: Math.max(1, pixelCount),
      selection,
      selectionRevision: state.selectionRevision + 1
    })),

  setSelectionMode: (mode) => set({ selectionMode: mode }),

  clearSelection: () =>
    set((state) => ({
      selectedPixel: null,
      selectedPixelCount: 0,
      selection: null,
      selectionRevision: state.selectionRevision + 1
    })),
  
  setStatus: (status, errorMessage = null) => set({ status, errorMessage }),
  
  setMatchOptions: (options) => set((state) => ({ 
    matchOptions: { ...state.matchOptions, ...options }
  })),

  setMatchData: (data) => set({ matchData: data }),

  toggleSelectedLibrarySpectrum: (item) =>
    set((state) => {
      const exists = state.selectedLibrarySpectra.some((entry) => entry.spectrum_id === item.spectrum_id)
      return {
        selectedLibrarySpectra: exists
          ? state.selectedLibrarySpectra.filter((entry) => entry.spectrum_id !== item.spectrum_id)
          : [...state.selectedLibrarySpectra, item]
      }
    }),

  addSelectedLibrarySpectra: (items) =>
    set((state) => {
      if (items.length === 0) {
        return { selectedLibrarySpectra: state.selectedLibrarySpectra }
      }
      const byId = new Map(state.selectedLibrarySpectra.map((item) => [item.spectrum_id, item]))
      items.forEach((item) => {
        byId.set(item.spectrum_id, item)
      })
      return {
        selectedLibrarySpectra: Array.from(byId.values())
      }
    }),

  removeSelectedLibrarySpectra: (spectrumIds) =>
    set((state) => {
      if (spectrumIds.length === 0) {
        return { selectedLibrarySpectra: state.selectedLibrarySpectra }
      }
      const removeSet = new Set(spectrumIds)
      return {
        selectedLibrarySpectra: state.selectedLibrarySpectra.filter((item) => !removeSet.has(item.spectrum_id))
      }
    }),

  clearSelectedLibrarySpectra: () =>
    set({
      selectedLibrarySpectra: [],
      selectedLibraryCurves: []
    }),

  setSelectedLibraryCurves: (items) => set({ selectedLibraryCurves: items })
}))
