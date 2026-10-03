import { useEffect, useState } from 'react'

export type ModelCatalog = {
  forecasting?: {
    model_family?: string
    horizons?: string[]
  }
  anomaly_detection?: {
    model?: string
    domain?: string
    supports_600mw?: boolean
  }
  attack_classification?: {
    model?: string
    domain?: string
    target_mechanisms?: number
  }
  decision_support?: {
    supported_domains?: string[]
    levels?: string[]
    '600mw_anomaly_state'?: string
  }
}

export type ApiSnapshot = {
  connection: 'checking' | 'online' | 'offline'
  models: ModelCatalog | null
  checkedAt: Date | null
}

const apiBase =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000'

async function getJson<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    signal,
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw new Error(`API returned ${response.status}`)
  return response.json() as Promise<T>
}

export function useApiSnapshot() {
  const [snapshot, setSnapshot] = useState<ApiSnapshot>({
    connection: 'checking',
    models: null,
    checkedAt: null,
  })

  useEffect(() => {
    let active = true
    let controller: AbortController | null = null

    const refresh = async () => {
      controller?.abort()
      controller = new AbortController()
      const timeout = window.setTimeout(() => controller?.abort(), 5000)
      const [healthResult, modelsResult] = await Promise.allSettled([
        getJson<{ status?: string }>('/health', controller.signal),
        getJson<ModelCatalog>('/models', controller.signal),
      ])
      window.clearTimeout(timeout)
      if (!active) return

      const healthy = healthResult.status === 'fulfilled' && healthResult.value.status === 'ok'
      setSnapshot({
        connection: healthy ? 'online' : 'offline',
        models: modelsResult.status === 'fulfilled' ? modelsResult.value : null,
        checkedAt: new Date(),
      })
    }

    void refresh()
    const interval = window.setInterval(() => void refresh(), 15000)
    return () => {
      active = false
      controller?.abort()
      window.clearInterval(interval)
    }
  }, [])

  return snapshot
}
