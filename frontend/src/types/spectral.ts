export interface SpectralMaskRange {
  start: number
  end: number
}

export interface QuerySpectrumData {
  x: number
  y: number
  bands_total: number
  bands_used: number
  selection_mode: 'pixel' | 'box' | 'circle' | 'lasso'
  selected_pixels: number
  wavelengths: number[]
  spectrum: number[]
}

export interface MatchResultItem {
  rank: number
  spectrum_id: number
  name: string
  class?: string | null
  sensor_type?: string | null
  measure_mode?: string | null
  source?: string | null
  sam_score: number
  pearson_r?: number | null
  curve?: number[] | null
}

export interface MatchContextData {
  signature_hash: string
  metric: string
  ignore_water_bands: boolean
  min_valid_bands: number
  custom_masked_ranges: SpectralMaskRange[]
  candidate_count: number
  elapsed_ms: number
}

export interface MatchData {
  query: QuerySpectrumData
  match_context: MatchContextData
  results: MatchResultItem[]
}

export interface LibrarySearchItem {
  spectrum_id: number
  name: string
  class?: string | null
  sensor_type?: string | null
  measure_mode?: string | null
  source?: string | null
  chapter?: string | null
}

export interface LibrarySearchData {
  query: string
  total: number
  items: LibrarySearchItem[]
}

export interface LibraryCurveItem extends LibrarySearchItem {
  wavelengths: number[]
  curve: number[]
}

export interface LibraryCurvesData {
  image_id: string
  wavelengths: number[]
  items: LibraryCurveItem[]
}
