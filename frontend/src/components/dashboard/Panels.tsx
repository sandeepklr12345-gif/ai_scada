import type { ApiSnapshot } from './api'

import type { EquipmentId, EquipmentSelection } from '../plant/equipment'

type Reading = { label: string; unit?: string }

const equipmentDetails: Record<EquipmentId, { title: string; readings: Reading[] }> = {
  CoalHandlingSystem: { title: 'Coal Handling', readings: [{ label: 'Coal flow', unit: 't/h' }, { label: 'AI status' }] },
  CoalMill: { title: 'Coal Mill', readings: [{ label: 'Coal flow', unit: 't/h' }, { label: 'AI status' }] },
  BoilerSystem: { title: 'Boiler', readings: [{ label: 'Steam temperature', unit: '°C' }, { label: 'Steam pressure', unit: 'MPa' }, { label: 'Drum pressure', unit: 'MPa' }, { label: 'Coal flow', unit: 't/h' }, { label: 'Feedwater flow', unit: 't/h' }, { label: 'AI status' }] },
  Chimney: { title: 'Chimney / Stack', readings: [{ label: 'Flue gas O₂', unit: '%' }, { label: 'Flue gas temperature', unit: '°C' }, { label: 'AI status' }] },
  TurbineSystem: { title: 'Turbine', readings: [{ label: 'Steam pressure', unit: 'MPa' }, { label: 'Steam temperature', unit: '°C' }, { label: 'Reheat steam flow', unit: 't/h' }, { label: 'Reheat temperature', unit: '°C' }, { label: 'AI status' }] },
  Generator: { title: 'Generator', readings: [{ label: 'Power output', unit: 'MW' }, { label: 'AI status' }] },
  CondenserSystem: { title: 'Condenser', readings: [{ label: 'Condenser vacuum', unit: 'bar' }, { label: 'Condenser temperature', unit: '°C' }, { label: 'Cooling water flow', unit: 'm³/h' }, { label: 'AI status' }] },
  FeedwaterSystem: { title: 'Feedwater System', readings: [{ label: 'Feedwater flow', unit: 't/h' }, { label: 'Feedwater pressure', unit: 'MPa' }, { label: 'AI status' }] },
  FeedwaterPump: { title: 'Feedwater Pump', readings: [{ label: 'Feedwater flow', unit: 't/h' }, { label: 'Feedwater pressure', unit: 'MPa' }, { label: 'AI status' }] },
  CoolingSystem: { title: 'Cooling System', readings: [{ label: 'Cooling water flow', unit: 'm³/h' }, { label: 'Condenser outlet temperature', unit: '°C' }, { label: 'Pump current', unit: 'A' }, { label: 'AI status' }] },
  ProcessPipes: { title: 'Process Piping', readings: [{ label: 'Main steam pressure', unit: 'MPa' }, { label: 'Feedwater flow', unit: 't/h' }, { label: 'AI status' }] },
}

function formatEquipment(selection: EquipmentSelection) {

  const details = equipmentDetails[selection.id]

  return selection.instance

    ? `${details.title} ${selection.instance}`

    : details.title

}

function formatNumber(value: number, decimals = 2) {

  return value.toFixed(decimals)

}

function getLatestPowerOutput(api: ApiSnapshot) {
  return api.runtime?.measurement?.value ?? null
}

function getRawFeatures(api: ApiSnapshot) {
  return api.raw600mw?.available ? api.raw600mw.features : {}
}

function normalizeFeatureName(name: string) {
  return name
    .normalize('NFKC')
    .replace(/\r?\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

function getRawValue(api: ApiSnapshot, name: string) {
  const target = normalizeFeatureName(name)
  const entry = Object.entries(getRawFeatures(api)).find(
    ([featureName]) => normalizeFeatureName(featureName) === target,
  )
  const value = entry?.[1]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getAverageRawValue(api: ApiSnapshot, names: string[]) {
  const values = names
    .map((name) => getRawValue(api, name))
    .filter((value): value is number => value !== null)
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null
}

function getTotalCoalFlow(api: ApiSnapshot) {
  const values = Object.entries(getRawFeatures(api))
    .filter(([name]) => /#\d+ coal mill .*output/i.test(normalizeFeatureName(name)))
    .map(([, value]) => value)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null
}

const RAW_NAMES = {
  steamPressure: 'Fresh steam pressure\n(MPa)',
  steamTemperatureA: 'A-side SH steam temperature\n(℃)',
  steamTemperatureB: 'B-side SH steam temperature\n(℃)',
  oxygenA: 'A-side flue gas oxygen content\n（%）',
  oxygenB: 'B-side flue gas oxygen content\n（%）',
  feedwaterFlow: 'Feed water flow rate\n（t/h）',
  feedwaterPressure: 'Feedwater pressure\n(MPa)',
  drumPressure: 'Drum pressure\n(MPa)',
  reheatSteamFlow: 'Reheat steam flow rate\n(t/h)',
  reheatTemperatureA: 'A-side reheater outlet steam temperature\n(℃)',
  reheatTemperatureB: 'B-side reheater outlet steam temperature\n(℃)',
  superheatedPressureA: 'A-side superheated steam pressure\n(MPa)',
  superheatedPressureB: 'B-side superheated steam pressure\n(MPa)',
  flueGasTemperatureA: 'A-side air preheater outlet flue gas temperature\n(℃)',
  flueGasTemperatureB: 'B-side air preheater outlet flue gas temperature\n(℃)',
}

function getCoalMillFlow(api: ApiSnapshot, instance?: number) {
  if (!instance) return getTotalCoalFlow(api)
  const a = getRawValue(api, `#${instance} coal mill A-side output\n（t/h）`)
  const b = getRawValue(api, `#${instance} coal mill B-side output\n（t/h）`)
  const values = [a, b].filter((value): value is number => value !== null)
  return values.length ? values.reduce((sum, value) => sum + value, 0) : null
}

function getEquipmentReadingValue(selection: EquipmentSelection, label: string, api: ApiSnapshot) {
  switch (selection.id) {
    case 'CoalHandlingSystem':
      return label === 'Coal flow' ? getTotalCoalFlow(api) : null
    case 'CoalMill':
      return label === 'Coal flow' ? getCoalMillFlow(api, selection.instance) : null
    case 'BoilerSystem':
      if (label === 'Steam temperature') return getAverageRawValue(api, [RAW_NAMES.steamTemperatureA, RAW_NAMES.steamTemperatureB])
      if (label === 'Steam pressure') return getRawValue(api, RAW_NAMES.steamPressure)
      if (label === 'Drum pressure') return getRawValue(api, RAW_NAMES.drumPressure)
      if (label === 'Coal flow') return getTotalCoalFlow(api)
      if (label === 'Feedwater flow') return getRawValue(api, RAW_NAMES.feedwaterFlow)
      return null
    case 'Chimney':
      if (label === 'Flue gas O₂') return getAverageRawValue(api, [RAW_NAMES.oxygenA, RAW_NAMES.oxygenB])
      if (label === 'Flue gas temperature') return getAverageRawValue(api, [RAW_NAMES.flueGasTemperatureA, RAW_NAMES.flueGasTemperatureB])
      return null
    case 'TurbineSystem':
      if (label === 'Steam temperature') return getAverageRawValue(api, [RAW_NAMES.steamTemperatureA, RAW_NAMES.steamTemperatureB])
      if (label === 'Steam pressure') return getAverageRawValue(api, [RAW_NAMES.superheatedPressureA, RAW_NAMES.superheatedPressureB])
      if (label === 'Reheat steam flow') return getRawValue(api, RAW_NAMES.reheatSteamFlow)
      if (label === 'Reheat temperature') return getAverageRawValue(api, [RAW_NAMES.reheatTemperatureA, RAW_NAMES.reheatTemperatureB])
      return null
    case 'Generator':
      return label === 'Power output' ? getLatestPowerOutput(api) : null
    case 'FeedwaterSystem':
    case 'FeedwaterPump':
      if (label === 'Feedwater flow') return getRawValue(api, RAW_NAMES.feedwaterFlow)
      if (label === 'Feedwater pressure') return getRawValue(api, RAW_NAMES.feedwaterPressure)
      return null
    case 'ProcessPipes':
      if (label === 'Main steam pressure') return getRawValue(api, RAW_NAMES.steamPressure)
      if (label === 'Feedwater flow') return getRawValue(api, RAW_NAMES.feedwaterFlow)
      return null
    default:
      return null
  }
}

function getForecast(api: ApiSnapshot, horizon: number) {

  return (

    api.predictions?.predictions.find(

      (prediction) =>

        prediction.prediction_horizon_minutes === horizon,

    ) ?? null

  )

}

export function EquipmentInspector({

  selection,

  api,

  onClose,

}: {

  selection: EquipmentSelection | null

  api: ApiSnapshot

  onClose: () => void

}) {

  if (!selection) return null

  const detail = equipmentDetails[selection.id]

  const anomalyUnavailable =

    api.models?.anomaly_detection?.supports_600mw === false

  const runtimeMeasurement = api.runtime?.measurement ?? null

  return (

    <section

      className="equipment-inspector"

      aria-live="polite"

      aria-label={`${detail.title} information`}

    >

      <div className="inspector-heading">

        <div>

          <span className="eyebrow">SELECTED EQUIPMENT</span>

          <h2>{formatEquipment(selection)}</h2>

        </div>

        <button

          type="button"

          className="icon-button"

          onClick={onClose}

          aria-label="Close equipment details"

        >

          ×

        </button>

      </div>

      <div className="inspector-readings">

        {detail.readings.map(({ label, unit }) => {

          const aiUnavailable =

            label === 'AI status' && anomalyUnavailable
          const numericValue = getEquipmentReadingValue(selection, label, api)
          let value = numericValue !== null ? formatNumber(numericValue) : 'N/A'


          if (aiUnavailable) {
            value = 'UNAVAILABLE'
          }

          return (

            <div

              className="inspector-reading"

              key={label}

            >

              <span>{label}</span>

              <strong

                className={

                  aiUnavailable ? 'value-warning' : ''

                }

              >

                {value}

                {unit ? <small> {unit}</small> : null}

              </strong>

            </div>

          )

        })}

      </div>

      <p className="panel-footnote">

        {runtimeMeasurement

          ? `Latest measurement: ${runtimeMeasurement.timestamp}`

          : 'Waiting for a connected SCADA measurement source.'}

      </p>

    </section>

  )

}

const overviewMetrics = [
  { label: 'Coal Handling', unit: 't/h', getValue: getTotalCoalFlow },
  { label: 'Coal Mill', unit: 't/h', getValue: getTotalCoalFlow },
  { label: 'Boiler', unit: '°C', getValue: (api: ApiSnapshot) => getAverageRawValue(api, [RAW_NAMES.steamTemperatureA, RAW_NAMES.steamTemperatureB]) },
  { label: 'Chimney', unit: '% O₂', getValue: (api: ApiSnapshot) => getAverageRawValue(api, [RAW_NAMES.oxygenA, RAW_NAMES.oxygenB]) },
  { label: 'Turbine', unit: '°C', getValue: (api: ApiSnapshot) => getAverageRawValue(api, [RAW_NAMES.steamTemperatureA, RAW_NAMES.steamTemperatureB]) },
  { label: 'Generator', unit: 'MW', getValue: getLatestPowerOutput },
  { label: 'Condenser', unit: '', getValue: () => null },
  { label: 'Feedwater System', unit: 't/h', getValue: (api: ApiSnapshot) => getRawValue(api, RAW_NAMES.feedwaterFlow) },
  { label: 'Feedwater Pump', unit: 't/h', getValue: (api: ApiSnapshot) => getRawValue(api, RAW_NAMES.feedwaterFlow) },
  { label: 'Cooling System', unit: '', getValue: () => null },
  { label: 'Process Piping', unit: 'MPa', getValue: (api: ApiSnapshot) => getRawValue(api, RAW_NAMES.steamPressure) },
]

export function PlantOverview({

  onMinimize,

  api,

}: {

  onMinimize: () => void

  api: ApiSnapshot

}) {

  const hasRuntimeData = api.raw600mw?.available === true

  return (

    <section

      className="rail-panel overview-panel"

      aria-labelledby="overview-title"

    >

      <div className="rail-panel-title">

        <span

          className="panel-icon panel-icon--plant"

          aria-hidden="true"

        >

          <svg

            viewBox="0 0 24 24"

            focusable="false"

          >

            <path d="M3 21h18M5 21V11h5v10m2 0V4h4v17m2 0V9h3v12M13 7h2m-2 3h2m-8 4h1m-1 3h1" />

          </svg>

        </span>

        <h2 id="overview-title">Plant Overview</h2>

        <button

          type="button"

          className="panel-minimize"

          onClick={onMinimize}

          aria-label="Minimize Plant Overview"

          title="Expand 3D view"

        >

          <svg

            viewBox="0 0 20 20"

            aria-hidden="true"

          >

            <path d="M3 6h14M3 14h14M7 3v14" />

          </svg>

        </button>

      </div>

      <div className="overview-banner">

        <div

          className="overview-schematic"

          aria-hidden="true"

        >

          <span className="schematic-stack" />

          <span className="schematic-boiler" />

          <span className="schematic-turbine" />

          <span className="schematic-tower" />

          <span className="schematic-pipe schematic-pipe-a" />

          <span className="schematic-pipe schematic-pipe-b" />

        </div>

        <div className="overview-copy">

          <strong>THERMAL POWER PLANT</strong>

          <small>INTERACTIVE DIGITAL TWIN</small>

          <span className="overview-waiting">

            <i aria-hidden="true" />

            {hasRuntimeData

              ? 'SCADA DATA CONNECTED'

              : 'WAITING FOR SCADA DATA'}

          </span>

        </div>

      </div>

      <div className="overview-metrics">

        {overviewMetrics.map(({ label, unit, getValue }, index) => {
          const metricValue = getValue(api)
          const value = metricValue !== null ? formatNumber(metricValue) : 'N/A'

          return (

            <div

              className="overview-metric"

              key={label}

            >

              <span

                className="metric-icon"

                aria-hidden="true"

              >

                {['ϟ', '↗', '♨', '⟳', '∿', '▧', '⇢'][index % 7]}

              </span>

              <span>{label}</span>

              <strong>

                {value}{unit ? ' ' : ''}

                {unit ? <small>{unit}</small> : null}

              </strong>

            </div>

          )

        })}

      </div>

      <p className="panel-footnote">

        {api.runtime?.measurement

          ? `Latest runtime reading: ${api.runtime.measurement.timestamp}`

          : 'Waiting for a connected SCADA measurement source.'}

      </p>

    </section>

  )

}

type StatusTone = 'good' | 'warning' | 'muted'

function SystemStatusRow({

  label,

  value,

  tone = 'muted',

}: {

  label: string

  value: string

  tone?: StatusTone

}) {

  return (

    <div className="system-status-row">

      <span

        className="status-symbol"

        aria-hidden="true"

      >

        ⌁

      </span>

      <span>{label}</span>

      <strong

        className={`status-badge status-badge--${tone}`}

      >

        {value}

      </strong>

    </div>

  )

}

export function AiSystemStatus({

  api,

}: {

  api: ApiSnapshot

}) {

  const models = api.models

  const offline = api.connection === 'offline'

  const forecast =

    offline

      ? 'API OFFLINE'

      : models?.forecasting

        ? api.predictions?.predictions?.length

          ? 'AVAILABLE · LIVE'

          : 'AVAILABLE · INPUT REQUIRED'

        : api.connection === 'checking'

          ? 'CHECKING'

          : 'NOT REPORTED'

  const anomaly =

    offline

      ? 'API OFFLINE'

      : models?.anomaly_detection?.supports_600mw === false

        ? 'UNAVAILABLE'

        : models?.anomaly_detection

          ? 'MODEL REPORTED'

          : api.connection === 'checking'

            ? 'CHECKING'

            : 'NOT REPORTED'

  const attack =

    offline

      ? 'API OFFLINE'

      : models?.attack_classification?.model

        ? 'MODEL REPORTED'

        : api.connection === 'checking'

          ? 'CHECKING'

          : 'NOT REPORTED'

  const decision =

    offline

      ? 'API OFFLINE'

      : models?.decision_support

        ? 'READY · WAITING INPUT'

        : api.connection === 'checking'

          ? 'CHECKING'

          : 'NOT REPORTED'

  return (

    <section

      className="rail-panel status-panel"

      aria-labelledby="ai-status-title"

    >

      <div className="rail-panel-title">

        <span

          className="panel-icon"

          aria-hidden="true"

        >

          ⌘

        </span>

        <h2 id="ai-status-title">

          AI System Status

        </h2>

      </div>

      <SystemStatusRow

        label="Forecast Model"

        value={forecast}

        tone={

          !offline && models?.forecasting

            ? 'good'

            : 'muted'

        }

      />

      <SystemStatusRow

        label="Anomaly Detection"

        value={anomaly}

        tone={

          models?.anomaly_detection

            ?.supports_600mw === false

            ? 'warning'

            : 'muted'

        }

      />

      <SystemStatusRow

        label="Attack Classifier"

        value={attack}

      />

      <SystemStatusRow

        label="Decision Engine"

        value={decision}

        tone={

          !offline && models?.decision_support

            ? 'good'

            : 'muted'

        }

      />

    </section>

  )

}

function parseDecisionDescription(description: string | null) {

  if (!description) {

    return {

      action: 'N/A',

      reason: 'No decision reason available.',

    }

  }

  const actionMatch = description.match(

    /action=([^;]+)/

  )

  const reasonMatch = description.match(

    /reason=(.*?); anomaly_score=/

  )

  return {

    action:

      actionMatch?.[1]

        ?.replaceAll('_', ' ')

        .trim() ?? 'N/A',

    reason:

      reasonMatch?.[1]?.trim() ??

      'No decision reason available.',

  }

}

export function RecentAlerts({

  api,

}: {

  api: ApiSnapshot

}) {

  const events = api.decisionHistory?.events ?? []

  const recentEvents = [...events]

    .sort(

      (a, b) =>

        new Date(b.timestamp).getTime() -

        new Date(a.timestamp).getTime(),

    )

    .slice(0, 5)

  return (

    <section

      className="rail-panel alerts-panel"

      aria-labelledby="alerts-title"

    >

      <div className="rail-panel-title alerts-heading">

        <span

          className="panel-icon"

          aria-hidden="true"

        >

          ◉

        </span>

        <h2 id="alerts-title">

          Recent Alerts

        </h2>

        <span className="rail-action">

          {recentEvents.length > 0

            ? `${recentEvents.length} EVENTS`

            : 'NO HISTORY'}

        </span>

      </div>

      {recentEvents.length === 0 ? (

        <div className="empty-alerts">

          <span

            className="empty-alert-icon"

            aria-hidden="true"

          >

            ?

          </span>

          <div>

            <strong>

              No decision events available

            </strong>

            <small>

              No DSS decision events have been

              reported by the API.

            </small>

          </div>

        </div>

      ) : (

        <div className="decision-alert-list">

          {recentEvents.map((event) => {

              const parsed = parseDecisionDescription(

                event.description,

              )

              const decisionLevel =

                event.alarm_code

                  ?.replace('DSS_', '')

                  .replaceAll('_', ' ') ??

                'N/A'

              return (

                <article

                  className="decision-alert"

                  key={event.event_id}

                >

                  <div className="decision-alert-heading">

                    <div>

                      <strong>

                        Decision Support Alert

                      </strong>

                      <small>

                        Event #{event.event_id}

                      </small>

                    </div>

                    <span

                      className={

                        event.severity === 'CRITICAL'

                          ? 'status-badge status-badge--warning'

                          : 'status-badge status-badge--muted'

                      }

                    >

                      {event.severity ?? 'UNKNOWN'}

                    </span>

                  </div>

                  <div className="decision-alert-grid">

                    <div>

                      <span>Decision Level</span>

                      <strong>

                        {decisionLevel}

                      </strong>

                    </div>

                    <div>

                      <span>Action</span>

                      <strong>

                        {parsed.action}

                      </strong>

                    </div>

                    <div>

                      <span>Status</span>

                      <strong>

                        {event.status

                          ?.replaceAll('_', ' ') ??

                          'N/A'}

                      </strong>

                    </div>

                    <div>

                      <span>Timestamp</span>

                      <strong>

                        {new Date(

                          event.timestamp,

                        ).toLocaleString()}

                      </strong>

                    </div>

                  </div>

                  <div className="decision-alert-reason">

                    <span>Decision Reason</span>

                    <p>

                      {parsed.reason}

                    </p>

                  </div>

                </article>

              )

            })}

        </div>

      )}

    </section>

  )

}

export function DecisionSupport({

  api,

}: {

  api: ApiSnapshot

}) {

  const anomalyUnsupported =

    api.models?.anomaly_detection

      ?.supports_600mw === false

  const engineReady =

    api.connection === 'online' &&

    Boolean(api.models?.decision_support)

  const powerOutput = getLatestPowerOutput(api)

  const forecast2 = getForecast(api, 2)

  const forecast10 = getForecast(api, 10)

  const forecast30 = getForecast(api, 30)

  const hasForecast =

    Boolean(forecast2 || forecast10 || forecast30)
  const latestDecision =

    api.decisionHistory?.events?.length

      ? [...api.decisionHistory.events].sort(

          (a, b) =>

            new Date(b.timestamp).getTime() -

            new Date(a.timestamp).getTime(),

        )[0]

      : null

  return (

    <section

      className="decision-support"

      aria-labelledby="decision-title"

    >

      <div className="decision-title-row">

        <span className="eyebrow">

          AI DECISION SUPPORT

        </span>

        <span

          className={`decision-state ${

            engineReady

              ? 'decision-state--ready'

              : ''

          }`}

        >

          {engineReady ? 'READY' : 'WAITING'}

        </span>

      </div>

      <h2 id="decision-title">

        {hasForecast

          ? 'Forecast available'

          : 'No active decision'}

      </h2>

      <div className="decision-reading">

        <span>Power Output</span>

        <strong>

          {powerOutput !== null

            ? `${formatNumber(powerOutput)} MW`

            : 'N/A'}

        </strong>

      </div>

      <div className="decision-reading">

        <span>2 min Forecast</span>

        <strong>

          {forecast2

            ? `${formatNumber(

                forecast2.predicted_value,

              )} MW`

            : 'N/A'}

        </strong>

      </div>

      <div className="decision-reading">

        <span>10 min Forecast</span>

        <strong>

          {forecast10

            ? `${formatNumber(

                forecast10.predicted_value,

              )} MW`

            : 'N/A'}

        </strong>

      </div>

      <div className="decision-reading">

        <span>30 min Forecast</span>

        <strong>

          {forecast30

            ? `${formatNumber(

                forecast30.predicted_value,

              )} MW`

            : 'N/A'}

        </strong>

      </div>

      <div className="decision-reading">

        <span>Anomaly Detection</span>

        <strong

          className={

            anomalyUnsupported

              ? 'value-warning'

              : ''

          }

        >

          {anomalyUnsupported

            ? 'UNAVAILABLE'

            : 'N/A'}

        </strong>

      </div>

      <div className="decision-level">

        <span>Latest DSS Decision</span>

        <strong>

          {latestDecision?.alarm_code

            ?.replace('DSS_', 'DSS ')

            .replaceAll('_', ' ') ??

            'N/A'}

        </strong>

      </div>

      <p className="panel-footnote">

        Forecast values are displayed from the

        live API. Latest DSS decision is shown from

        persisted decision history.

      </p>

    </section>

  )

}

const liveParameters = [
  { label: 'Output Power', unit: 'MW', getValue: getLatestPowerOutput },
  { label: 'Steam Pressure', unit: 'MPa', getValue: (api: ApiSnapshot) => getRawValue(api, RAW_NAMES.steamPressure) },
  { label: 'Steam Temperature', unit: '°C', getValue: (api: ApiSnapshot) => getAverageRawValue(api, [RAW_NAMES.steamTemperatureA, RAW_NAMES.steamTemperatureB]) },
  { label: 'Flue Gas O₂', unit: '%', getValue: (api: ApiSnapshot) => getAverageRawValue(api, [RAW_NAMES.oxygenA, RAW_NAMES.oxygenB]) },
  { label: 'Feedwater Flow', unit: 't/h', getValue: (api: ApiSnapshot) => getRawValue(api, RAW_NAMES.feedwaterFlow) },
  { label: 'Total Coal Flow', unit: 't/h', getValue: getTotalCoalFlow },
]

export function LiveParameters({
  api,
}: {
  api: ApiSnapshot
}) {
  const hasRuntime = api.raw600mw?.available === true

  return (
    <section className="content-section live-section" id="live-parameters" aria-labelledby="live-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">PLANT TELEMETRY</span>
          <h2 id="live-title">Main Plant Parameters</h2>
        </div>
        <span className={`section-status status-badge ${hasRuntime ? 'status-badge--good' : 'status-badge--muted'}`}>
          {hasRuntime ? 'SCADA DATA CONNECTED' : 'WAITING FOR SCADA DATA'}
        </span>
      </div>
      <div className="parameter-grid">
        {liveParameters.map(({ label, unit, getValue }) => {
          const value = getValue(api)
          return (
            <article className="parameter-card" key={label}>
              <span className="parameter-label">{label}</span>
              <strong>{value !== null ? formatNumber(value) : 'N/A'} <small>{unit}</small></strong>
              <span className="parameter-note">
                {value !== null ? 'Historical 600 MW replay · RAW_SOURCE' : 'Awaiting telemetry'}
              </span>
            </article>
          )
        })}
      </div>
      {api.raw600mw?.timestamp && (
        <p className="panel-footnote">
          Source: {api.raw600mw.source ?? '600MW_HISTORICAL_REPLAY'} · Step: {api.raw600mw.step ?? 'N/A'} · Timestamp: {new Date(api.raw600mw.timestamp).toLocaleString()}
        </p>
      )}
    </section>
  )
}

function ForecastCard({

  title,

  prediction,

}: {

  title: string

  prediction: ReturnType<typeof getForecast>

}) {

  return (

    <article className="analytics-card">

      <div className="chart-heading">

        <h3>{title}</h3>

        <span>MW</span>

      </div>

      <div className="decision-empty">

        <strong>

          {prediction

            ? `${formatNumber(

                prediction.predicted_value,

              )} MW`

            : 'N/A'}

        </strong>

        <p>

          {prediction

            ? `Forecast horizon: ${prediction.prediction_horizon_minutes} minutes`

            : 'Forecast not available from the API.'}

        </p>

      </div>

    </article>

  )

}

const charts = [

  {

    title: 'Power Output',

    unit: 'MW',

    message: 'Waiting for SCADA time series.',

  },

  {

    title: 'Steam Temperature',

    unit: '°C',

    message: 'Waiting for SCADA time series.',

  },

  {

    title: 'Steam Pressure',

    unit: 'MPa',

    message: 'Waiting for SCADA time series.',

  },

  {

    title: 'Forecast vs Actual',

    unit: 'MW',

    message:

      'Forecast values are shown in the live forecast cards below.',

  },

  {

    title: 'Turbine Speed',

    unit: 'RPM',

    message: 'Waiting for SCADA time series.',

  },

  {

    title: 'Coal Flow',

    unit: 't/h',

    message: 'Waiting for SCADA time series.',

  },

  {

    title: 'Feedwater Flow',

    unit: 't/h',

    message: 'Waiting for SCADA time series.',

  },

]

function EmptyChart({

  title,

  unit,

  message,

}: {

  title: string

  unit: string

  message: string

}) {

  return (

    <article className="analytics-card">

      <div className="chart-heading">

        <h3>{title}</h3>

        <span>{unit}</span>

      </div>

      <div className="chart-empty">

        <svg

          viewBox="0 0 320 112"

          role="img"

          aria-label={`${title}: no connected data`}

        >

          {[16, 40, 64, 88].map((y) => (

            <line

              key={y}

              x1="0"

              x2="320"

              y1={y}

              y2={y}

            />

          ))}

          {[40, 100, 160, 220, 280].map(

            (x) => (

              <line

                key={x}

                x1={x}

                x2={x}

                y1="8"

                y2="104"

              />

            ),

          )}

        </svg>

        <p>{message}</p>

      </div>

    </article>

  )

}

export function Analytics({

  api,

}: {

  api: ApiSnapshot

}) {

  const forecast2 = getForecast(api, 2)

  const forecast10 = getForecast(api, 10)

  const forecast30 = getForecast(api, 30)

  const hasForecast =

    Boolean(forecast2 || forecast10 || forecast30)

  const decisionEvents = [

    ...(api.decisionHistory?.events ?? []),

  ]

    .sort(

      (a, b) =>

        new Date(b.timestamp).getTime() -

        new Date(a.timestamp).getTime(),

    )

    .slice(0, 5)

  return (

    <section

      className="content-section analytics-section"

      id="analytics"

      aria-labelledby="analytics-title"

    >

      <div className="section-heading">

        <div>

          <span className="eyebrow">

            TRENDS & FORECASTS

          </span>

          <h2 id="analytics-title">

            Analytics &amp; Graphs

          </h2>

        </div>

        <span

          className={`section-status status-badge ${

            hasForecast

              ? 'status-badge--good'

              : 'status-badge--muted'

          }`}

        >

          {hasForecast

            ? 'FORECAST DATA AVAILABLE'

            : 'NO SERIES AVAILABLE'}

        </span>

      </div>

      <div className="analytics-grid">

        {charts.map((chart) => (

          <EmptyChart

            key={chart.title}

            {...chart}

          />

        ))}

        <ForecastCard

          title="2 Minute Forecast"

          prediction={forecast2}

        />

        <ForecastCard

          title="10 Minute Forecast"

          prediction={forecast10}

        />

        <ForecastCard

          title="30 Minute Forecast"

          prediction={forecast30}

        />

        <article className="analytics-card decision-history">

          <div className="chart-heading">

            <h3>Decision Support</h3>

            <span>EVENTS</span>

          </div>

          {decisionEvents.length === 0 ? (

            <div className="decision-empty">

              <strong>

                No decision history available

              </strong>

              <p>

                No persisted DSS decision events have

                been reported by the API.

              </p>

            </div>

          ) : (

            <div className="decision-history-list">

              {decisionEvents.map((event) => (

                <article

                  className="decision-history-item"

                  key={event.event_id}

                >

                  <strong>

                    {event.alarm_code ??

                      event.event_type}

                  </strong>

                  <span>

                    {event.status ?? 'UNKNOWN'}

                  </span>

                  <small>

                    {new Date(

                      event.timestamp,

                    ).toLocaleString()}

                  </small>

                </article>

              ))}

            </div>

          )}

        </article>

      </div>

    </section>

  )

}