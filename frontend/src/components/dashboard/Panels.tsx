import type { ApiSnapshot } from './api'
import type { EquipmentId, EquipmentSelection } from '../plant/equipment'

type Reading = { label: string; unit?: string }

const equipmentDetails: Record<EquipmentId, { title: string; readings: Reading[] }> = {
  CoalHandlingSystem: {
    title: 'Coal Handling',
    readings: [{ label: 'Coal flow', unit: 't/h' }, { label: 'Storage level', unit: '%' }, { label: 'Transfer status' }, { label: 'AI status' }],
  },
  CoalMill: {
    title: 'Coal Mill',
    readings: [{ label: 'Coal flow', unit: 't/h' }, { label: 'Mill current', unit: 'A' }, { label: 'AI status' }],
  },
  BoilerSystem: {
    title: 'Boiler',
    readings: [
      { label: 'Steam temperature', unit: '°C' },
      { label: 'Steam pressure', unit: 'MPa' },
      { label: 'Drum level', unit: '%' },
      { label: 'Coal flow', unit: 't/h' },
      { label: 'Feedwater flow', unit: 't/h' },
      { label: 'AI status' },
    ],
  },
  Chimney: {
    title: 'Chimney / Stack',
    readings: [{ label: 'Flue gas temperature', unit: '°C' }, { label: 'Stack draft', unit: 'Pa' }, { label: 'AI status' }],
  },
  TurbineSystem: {
    title: 'Turbine',
    readings: [
      { label: 'Turbine speed', unit: 'RPM' },
      { label: 'Inlet steam temperature', unit: '°C' },
      { label: 'Inlet steam pressure', unit: 'MPa' },
      { label: 'Vibration X', unit: 'mm/s' },
      { label: 'Vibration Y', unit: 'mm/s' },
      { label: 'AI status' },
    ],
  },
  Generator: {
    title: 'Generator',
    readings: [
      { label: 'Power output', unit: 'MW' },
      { label: 'Voltage', unit: 'kV' },
      { label: 'Current', unit: 'kA' },
      { label: 'Frequency', unit: 'Hz' },
      { label: 'AI status' },
    ],
  },
  CondenserSystem: {
    title: 'Condenser',
    readings: [
      { label: 'Condenser vacuum', unit: 'bar' },
      { label: 'Condenser temperature', unit: '°C' },
      { label: 'Cooling water flow', unit: 'm³/h' },
      { label: 'AI status' },
    ],
  },
  FeedwaterSystem: {
    title: 'Feedwater System',
    readings: [{ label: 'Feedwater flow', unit: 't/h' }, { label: 'Discharge pressure', unit: 'MPa' }, { label: 'Motor current', unit: 'A' }, { label: 'AI status' }],
  },
  FeedwaterPump: {
    title: 'Feedwater Pump',
    readings: [{ label: 'Feedwater flow', unit: 't/h' }, { label: 'Discharge pressure', unit: 'MPa' }, { label: 'Motor current', unit: 'A' }, { label: 'AI status' }],
  },
  CoolingSystem: {
    title: 'Cooling System',
    readings: [{ label: 'Cooling water flow', unit: 'm³/h' }, { label: 'Condenser outlet temperature', unit: '°C' }, { label: 'Pump current', unit: 'A' }, { label: 'AI status' }],
  },
  ProcessPipes: {
    title: 'Process Piping',
    readings: [{ label: 'Main steam pressure', unit: 'MPa' }, { label: 'Feedwater flow', unit: 't/h' }, { label: 'Cooling water flow', unit: 'm³/h' }, { label: 'AI status' }],
  },
}

function formatEquipment(selection: EquipmentSelection) {
  const details = equipmentDetails[selection.id]
  return selection.instance ? `${details.title} ${selection.instance}` : details.title
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
  const anomalyUnavailable = api.models?.anomaly_detection?.supports_600mw === false

  return (
    <section className="equipment-inspector" aria-live="polite" aria-label={`${detail.title} information`}>
      <div className="inspector-heading">
        <div>
          <span className="eyebrow">SELECTED EQUIPMENT</span>
          <h2>{formatEquipment(selection)}</h2>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close equipment details">×</button>
      </div>
      <div className="inspector-readings">
        {detail.readings.map(({ label, unit }) => {
          const aiUnavailable = label === 'AI status' && anomalyUnavailable
          const value = aiUnavailable ? 'UNAVAILABLE' : 'N/A'
          return (
            <div className="inspector-reading" key={label}>
              <span>{label}</span>
              <strong className={aiUnavailable ? 'value-warning' : ''}>{value}{unit ? <small> {unit}</small> : null}</strong>
            </div>
          )
        })}
      </div>
      <p className="panel-footnote">The current API does not expose live equipment measurements.</p>
    </section>
  )
}

const overviewMetrics: Reading[] = [
  { label: 'Power output', unit: 'MW' },
  { label: 'Main steam pressure', unit: 'MPa' },
  { label: 'Main steam temperature', unit: '°C' },
  { label: 'Turbine speed', unit: 'RPM' },
  { label: 'Generator frequency', unit: 'Hz' },
  { label: 'Total coal flow', unit: 't/h' },
  { label: 'Feedwater flow', unit: 't/h' },
]

export function PlantOverview({ onMinimize }: { onMinimize: () => void }) {
  return (
    <section className="rail-panel overview-panel" aria-labelledby="overview-title">
      <div className="rail-panel-title">
        <span className="panel-icon panel-icon--plant" aria-hidden="true">
          <svg viewBox="0 0 24 24" focusable="false"><path d="M3 21h18M5 21V11h5v10m2 0V4h4v17m2 0V9h3v12M13 7h2m-2 3h2m-8 4h1m-1 3h1" /></svg>
        </span>
        <h2 id="overview-title">Plant Overview</h2>
        <button type="button" className="panel-minimize" onClick={onMinimize} aria-label="Minimize Plant Overview" title="Expand 3D view">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M3 6h14M3 14h14M7 3v14" /></svg>
        </button>
      </div>
      <div className="overview-banner">
        <div className="overview-schematic" aria-hidden="true">
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
          <span className="overview-waiting"><i aria-hidden="true" />WAITING FOR SCADA DATA</span>
        </div>
      </div>
      <div className="overview-metrics">
        {overviewMetrics.map(({ label, unit }, index) => (
          <div className="overview-metric" key={label}>
            <span className="metric-icon" aria-hidden="true">{['ϟ', '↗', '♨', '⟳', '∿', '▧', '⇢'][index]}</span>
            <span>{label}</span>
            <strong>N/A <small>{unit}</small></strong>
          </div>
        ))}
      </div>
      <p className="panel-footnote">Waiting for a connected SCADA measurement source.</p>
    </section>
  )
}

type StatusTone = 'good' | 'warning' | 'muted'

function SystemStatusRow({ label, value, tone = 'muted' }: { label: string; value: string; tone?: StatusTone }) {
  return (
    <div className="system-status-row">
      <span className="status-symbol" aria-hidden="true">⌁</span>
      <span>{label}</span>
      <strong className={`status-badge status-badge--${tone}`}>{value}</strong>
    </div>
  )
}

export function AiSystemStatus({ api }: { api: ApiSnapshot }) {
  const models = api.models
  const offline = api.connection === 'offline'
  const forecast = offline ? 'API OFFLINE' : models?.forecasting ? 'AVAILABLE · INPUT REQUIRED' : api.connection === 'checking' ? 'CHECKING' : 'NOT REPORTED'
  const anomaly = offline ? 'API OFFLINE' : models?.anomaly_detection?.supports_600mw === false ? 'UNAVAILABLE' : models?.anomaly_detection ? 'MODEL REPORTED' : api.connection === 'checking' ? 'CHECKING' : 'NOT REPORTED'
  const attack = offline ? 'API OFFLINE' : models?.attack_classification?.model ? 'MODEL REPORTED' : api.connection === 'checking' ? 'CHECKING' : 'NOT REPORTED'
  const decision = offline ? 'API OFFLINE' : models?.decision_support ? 'READY · WAITING INPUT' : api.connection === 'checking' ? 'CHECKING' : 'NOT REPORTED'

  return (
    <section className="rail-panel status-panel" aria-labelledby="ai-status-title">
      <div className="rail-panel-title"><span className="panel-icon" aria-hidden="true">⌘</span><h2 id="ai-status-title">AI System Status</h2></div>
      <SystemStatusRow label="Forecast Model" value={forecast} tone={!offline && models?.forecasting ? 'good' : 'muted'} />
      <SystemStatusRow label="Anomaly Detection" value={anomaly} tone={models?.anomaly_detection?.supports_600mw === false ? 'warning' : 'muted'} />
      <SystemStatusRow label="Attack Classifier" value={attack} />
      <SystemStatusRow label="Decision Engine" value={decision} tone={!offline && models?.decision_support ? 'good' : 'muted'} />
    </section>
  )
}

export function RecentAlerts() {
  return (
    <section className="rail-panel alerts-panel" aria-labelledby="alerts-title">
      <div className="rail-panel-title alerts-heading">
        <span className="panel-icon" aria-hidden="true">◉</span>
        <h2 id="alerts-title">Recent Alerts</h2>
        <span className="rail-action">NO HISTORY</span>
      </div>
      <div className="empty-alerts">
        <span className="empty-alert-icon" aria-hidden="true">?</span>
        <div><strong>No alert history available</strong><small>The current API does not expose an alert history feed.</small></div>
      </div>
    </section>
  )
}

export function DecisionSupport({ api }: { api: ApiSnapshot }) {
  const anomalyUnsupported = api.models?.anomaly_detection?.supports_600mw === false
  const engineReady = api.connection === 'online' && Boolean(api.models?.decision_support)
  return (
    <section className="decision-support" aria-labelledby="decision-title">
      <div className="decision-title-row"><span className="eyebrow">AI DECISION SUPPORT</span><span className={`decision-state ${engineReady ? 'decision-state--ready' : ''}`}>{engineReady ? 'READY' : 'WAITING'}</span></div>
      <h2 id="decision-title">No active decision</h2>
      <div className="decision-reading"><span>Power Output</span><strong>N/A</strong></div>
      <div className="decision-reading"><span>Forecast</span><strong>N/A</strong></div>
      <div className="decision-reading"><span>Anomaly Detection</span><strong className={anomalyUnsupported ? 'value-warning' : ''}>{anomalyUnsupported ? 'UNAVAILABLE' : 'N/A'}</strong></div>
      <div className="decision-level"><span>Decision Level</span><strong>N/A</strong></div>
      <p className="panel-footnote">Decision results appear after supported input reaches the API.</p>
    </section>
  )
}

const liveParameters: Reading[] = [
  { label: 'Power Output', unit: 'MW' },
  { label: 'Steam Pressure', unit: 'MPa' },
  { label: 'Steam Temperature', unit: '°C' },
  { label: 'Turbine Speed', unit: 'RPM' },
  { label: 'Feedwater Flow', unit: 't/h' },
  { label: 'Coal Flow', unit: 't/h' },
]

export function LiveParameters() {
  return (
    <section className="content-section live-section" id="live-parameters" aria-labelledby="live-title">
      <div className="section-heading">
        <div><span className="eyebrow">PLANT TELEMETRY</span><h2 id="live-title">Live Parameters</h2></div>
        <span className="section-status status-badge status-badge--muted">SCADA FEED NOT EXPOSED</span>
      </div>
      <div className="parameter-grid">
        {liveParameters.map(({ label, unit }) => (
          <article className="parameter-card" key={label}>
            <span className="parameter-label">{label}</span>
            <strong>N/A <small>{unit}</small></strong>
            <span className="parameter-note">Awaiting measurement</span>
          </article>
        ))}
      </div>
    </section>
  )
}

const charts = [
  { title: 'Power Output', unit: 'MW', message: 'Waiting for SCADA time series.' },
  { title: 'Steam Temperature', unit: '°C', message: 'Waiting for SCADA time series.' },
  { title: 'Steam Pressure', unit: 'MPa', message: 'Waiting for SCADA time series.' },
  { title: 'Forecast vs Actual', unit: 'MW', message: 'Forecast requires supported feature input.' },
  { title: 'Turbine Speed', unit: 'RPM', message: 'Waiting for SCADA time series.' },
  { title: 'Coal Flow', unit: 't/h', message: 'Waiting for SCADA time series.' },
  { title: 'Feedwater Flow', unit: 't/h', message: 'Waiting for SCADA time series.' },
]

function EmptyChart({ title, unit, message }: { title: string; unit: string; message: string }) {
  return (
    <article className="analytics-card">
      <div className="chart-heading"><h3>{title}</h3><span>{unit}</span></div>
      <div className="chart-empty">
        <svg viewBox="0 0 320 112" role="img" aria-label={`${title}: no connected data`}>
          {[16, 40, 64, 88].map((y) => <line key={y} x1="0" x2="320" y1={y} y2={y} />)}
          {[40, 100, 160, 220, 280].map((x) => <line key={x} x1={x} x2={x} y1="8" y2="104" />)}
        </svg>
        <p>{message}</p>
      </div>
    </article>
  )
}

export function Analytics() {
  return (
    <section className="content-section analytics-section" id="analytics" aria-labelledby="analytics-title">
      <div className="section-heading">
        <div><span className="eyebrow">TRENDS & FORECASTS</span><h2 id="analytics-title">Analytics &amp; Graphs</h2></div>
        <span className="section-status status-badge status-badge--muted">NO SERIES AVAILABLE</span>
      </div>
      <div className="analytics-grid">
        {charts.map((chart) => <EmptyChart key={chart.title} {...chart} />)}
        <article className="analytics-card decision-history">
          <div className="chart-heading"><h3>Decision Support</h3><span>EVENTS</span></div>
          <div className="decision-empty"><strong>No decision history available</strong><p>The current API provides decision results on request, but no event-history feed.</p></div>
        </article>
      </div>
    </section>
  )
}
