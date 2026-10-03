const API_BASE_URL = 'http://127.0.0.1:8000'

export interface HealthResponse {
  status: string
  service: string
}

export interface ModelsResponse {
  forecasting: {
    model_family: string
    horizons: string[]
  }
  anomaly_detection: {
    model: string
    domain: string
    features: number
    supports_600mw: boolean
  }
  attack_classification: {
    model: string
    domain: string
    features: number
    target_mechanisms: number
  }
  decision_support: {
    supported_domains: string[]
    '600mw_anomaly_state': string
    levels: string[]
  }
}

async function apiRequest<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`)
  }

  return response.json() as Promise<T>
}

export async function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>('/health')
}

export async function getModels(): Promise<ModelsResponse> {
  return apiRequest<ModelsResponse>('/models')
}