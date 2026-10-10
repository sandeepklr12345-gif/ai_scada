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

export type RuntimeMeasurement = {
  measurement_id: number
  timestamp: string
  source_id: number
  equipment_id: number
  parameter_id: number
  value: number
  quality_status: string
}

export type RuntimeResponse = {
  status: string
  measurement: RuntimeMeasurement
}

export type Prediction = {
  prediction_id: number
  timestamp: string
  source_id: number
  equipment_id: number
  model_name: string
  prediction_type: string
  predicted_value: number
  confidence: number | null
  risk_level: string | null
  prediction_horizon_minutes: number
  description: string
}

export type PredictionsResponse = {
  status: string
  predictions: Prediction[]
}

export type DecisionEvent = {
  event_id: number
  timestamp: string
  source_id: number
  equipment_id: number
  event_type: string
  alarm_code: string
  severity: string
  status: string
  description: string
}

export type DecisionHistoryResponse = {
  status: string
  events: DecisionEvent[]
}

export type Raw600MwResponse = {
  status: string
  available: boolean
  source?: string
  timestamp?: string
  step?: number
  feature_count?: number
  features: Record<string, number>
}

export type ApiSnapshot = {
  connection: 'checking' | 'online' | 'offline'
  models: ModelCatalog | null
  runtime: RuntimeResponse | null
  predictions: PredictionsResponse | null
  decisionHistory: DecisionHistoryResponse | null
  raw600mw: Raw600MwResponse | null
  checkedAt: Date | null
}

const apiBase =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000'

async function getJson<T>(
  path: string,
  signal: AbortSignal,
): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, {
    signal,
    headers: {
      Accept: 'application/json',
    },
  })

  if (!response.ok) {
    throw new Error(`API returned ${response.status}`)
  }

  return response.json() as Promise<T>
}

export function useApiSnapshot() {
  const [snapshot, setSnapshot] = useState<ApiSnapshot>({
    connection: 'checking',
    models: null,
    runtime: null,
    predictions: null,
    decisionHistory: null,
    raw600mw: null,
    checkedAt: null,
  })

  useEffect(() => {
    let active = true
    let controller: AbortController | null = null

    const refresh = async () => {
      controller?.abort()
      controller = new AbortController()

      const timeout = window.setTimeout(() => {
        controller?.abort()
      }, 5000)

      const [
        healthResult,
        modelsResult,
        runtimeResult,
        predictionsResult,
        decisionHistoryResult,
        raw600mwResult,
      ] = await Promise.allSettled([
        getJson<{ status?: string }>(
          '/health',
          controller.signal,
        ),

        getJson<ModelCatalog>(
          '/models',
          controller.signal,
        ),

        getJson<RuntimeResponse>(
          '/runtime/latest',
          controller.signal,
        ),

        getJson<PredictionsResponse>(
          '/predictions/latest',
          controller.signal,
        ),

        getJson<DecisionHistoryResponse>(
          '/decision/history',
          controller.signal,
        ),

        getJson<Raw600MwResponse>(
          '/runtime/raw',
          controller.signal,
        ),
      ])

      window.clearTimeout(timeout)

      if (!active) {
        return
      }

      const healthy =
        healthResult.status === 'fulfilled' &&
        healthResult.value.status === 'ok'

      setSnapshot({
        connection: healthy ? 'online' : 'offline',

        models:
          modelsResult.status === 'fulfilled'
            ? modelsResult.value
            : null,

        runtime:
          runtimeResult.status === 'fulfilled'
            ? runtimeResult.value
            : null,

        predictions:
          predictionsResult.status === 'fulfilled'
            ? predictionsResult.value
            : null,

        decisionHistory:
          decisionHistoryResult.status === 'fulfilled'
            ? decisionHistoryResult.value
            : null,

        raw600mw:
          raw600mwResult.status === 'fulfilled'
            ? raw600mwResult.value
            : null,

        checkedAt: new Date(),
      })
    }

    void refresh()

    const interval = window.setInterval(() => {
      void refresh()
    }, 1000)

    return () => {
      active = false
      controller?.abort()
      window.clearInterval(interval)
    }
  }, [])

  return snapshot
}