import { useMutation, useQuery } from '@tanstack/react-query'
import apiClient from './client'

// Load Local Image
export const useLoadImageMutation = () => {
  return useMutation({
    mutationFn: async (payload: { image_path: string; display_mode: string; build_signature_cache: boolean; ignore_water_bands: boolean }) => {
      const { data } = await apiClient.post('/image/load', payload)
      if (data.code !== 0) throw new Error(data.message)
      return data.data
    }
  })
}

// Check Cache Signature Status
export const useSignatureStatusQuery = (signatureHash: string | null, enabled: boolean) => {
  return useQuery({
    queryKey: ['signatureStatus', signatureHash],
    queryFn: async () => {
      if (!signatureHash) return null
      const { data } = await apiClient.get(`/cache/signature/${signatureHash}/status`)
      if (data.code !== 0) throw new Error(data.message)
      return data.data
    },
    enabled: !!signatureHash && enabled,
    refetchInterval: (data: any) => {
      return data?.status === 'building' ? 1000 : false
    }
  })
}

// Match Pixel
export const useMatchPixelMutation = () => {
  return useMutation({
    mutationFn: async (payload: {
      image_id: string;
      x: number;
      y: number;
      top_n: number;
      metric: string;
      ignore_water_bands: boolean;
      min_valid_bands?: number;
      return_candidate_curves: boolean;
      selection?: any;
      custom_masked_ranges?: Array<{ start: number; end: number }>;
    }) => {
      const { data } = await apiClient.post('/match/pixel', payload)
      if (data.code !== 0) throw new Error(data.message)
      return data.data
    }
  })
}

// Export Match Result
export const useExportResultMutation = () => {
  return useMutation({
    mutationFn: async (payload: {
      image_id: string;
      x: number;
      y: number;
      top_n: number;
      format: string;
      output_path: string;
      include_query_spectrum: boolean;
      include_matched_curves: boolean;
      ignore_water_bands: boolean;
      min_valid_bands?: number;
      selection?: any;
    }) => {
      const { data } = await apiClient.post('/export/match-result', payload)
      if (data.code !== 0) throw new Error(data.message)
      return data.data
    }
  })
}
