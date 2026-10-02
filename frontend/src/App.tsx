import { useState } from 'react'
import './App.css'
import { useApiSnapshot } from './components/dashboard/api'
import {
  AiSystemStatus,
  Analytics,
  DecisionSupport,
  EquipmentInspector,
  LiveParameters,
  PlantOverview,
  RecentAlerts,
} from './components/dashboard/Panels'
import ThermalPowerPlant from './components/plant/ThermalPowerPlant'
import type { EquipmentSelection } from './components/plant/equipment'

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function App() {
  const api = useApiSnapshot()
  const [selection, setSelection] = useState<EquipmentSelection | null>(null)
  const [overviewCollapsed, setOverviewCollapsed] = useState(false)
  const apiStatus = {
    checking: 'CHECKING',
    online: 'ONLINE',
    offline: 'OFFLINE',
  }[api.connection]

  return (
    <main className="app">
      <header className="topbar">
        <button className="brand-lockup" onClick={() => scrollToSection('plant')} aria-label="AI SCADA Plant View">
          <svg className="brand-mark" viewBox="0 0 36 36" aria-hidden="true">
            <path d="M4 31h28M7 31V18h7v13M17 31V8h6v23M25 31V14h5v17M18 12h4M18 16h4M9 22h3M9 26h3" />
            <path d="M19 5h2" />
          </svg>
          <span className="brand-name">AI_SCADA</span>
          <span className="brand-divider" />
          <span className="brand-subtitle">AI Powered Thermal Power Plant Monitoring &amp; Decision Support</span>
        </button>

        <nav className="top-navigation" aria-label="Dashboard sections">
          <button className="nav-tab nav-tab--active" onClick={() => scrollToSection('plant')}>Plant View (3D)</button>
          <button className="nav-tab" onClick={() => scrollToSection('analytics')}>Analytics &amp; Graphs</button>
        </nav>

        <div className="connection-group" aria-label="Service connection status">
          <span className="connection-chip" title="No live measurement feed is exposed by the current API">
            <i className="connection-indicator connection-indicator--offline" />
            <span>LIVE</span><strong>NO FEED</strong>
          </span>
          <span className="connection-chip" title="MQTT connection status is not exposed to the frontend">
            <i className="connection-indicator connection-indicator--offline" />
            <span>MQTT</span><strong>N/A</strong>
          </span>
          <span className="connection-chip" title={api.checkedAt ? `Health endpoint checked ${api.checkedAt.toLocaleTimeString()}` : 'Checking API health'}>
            <i className={`connection-indicator connection-indicator--${api.connection}`} />
            <span>API</span><strong>{apiStatus}</strong>
          </span>
          <span className="connection-chip" title="Database connection status is not exposed to the frontend">
            <i className="connection-indicator connection-indicator--offline" />
            <span className="connection-name--full">DATABASE</span><span className="connection-name--short">DB</span><strong>N/A</strong>
          </span>
        </div>
      </header>

      <section className="plant-screen" id="plant" aria-label="Interactive thermal plant and status overview">
        <div className={`plant-layout${overviewCollapsed ? ' plant-layout--overview-collapsed' : ''}`}>
          <div className="plant-stage">
            <div className="stage-heading">
              <div>
                <span className="eyebrow">LIVE DIGITAL TWIN</span>
                <h1>THERMAL POWER PLANT</h1>
                <p>Interactive 3D equipment view</p>
              </div>
              <div className="plant-controls" aria-label="3D controls">
                <span>DRAG ROTATE</span><span>CTRL + WHEEL ZOOM</span><span>RIGHT-DRAG PAN</span>
              </div>
            </div>
            <div className="plant-view">
              <ThermalPowerPlant selection={selection} onSelectionChange={setSelection} />
            </div>
            <div className="scene-legend" aria-label="Process flow colors">
              <span><i className="flow-swatch flow-swatch--steam" />Steam</span>
              <span><i className="flow-swatch flow-swatch--feedwater" />Feedwater</span>
              <span><i className="flow-swatch flow-swatch--cooling" />Cooling water</span>
              <span><i className="flow-swatch flow-swatch--coal" />Coal</span>
            </div>
            <DecisionSupport api={api} />
            <EquipmentInspector selection={selection} api={api} onClose={() => setSelection(null)} />
            {overviewCollapsed && (
              <button
                type="button"
                className="overview-restore"
                onClick={() => setOverviewCollapsed(false)}
                aria-label="Restore Plant Overview panel"
              >
                <span className="restore-icon" aria-hidden="true">›</span>
                <span><strong>PLANT OVERVIEW</strong><small>Show status panel</small></span>
              </button>
            )}
          </div>

          <aside className={`plant-sidebar${overviewCollapsed ? ' plant-sidebar--collapsed' : ''}`} aria-label="Plant and AI status panels" aria-hidden={overviewCollapsed}>
            <PlantOverview onMinimize={() => setOverviewCollapsed(true)} />
            <AiSystemStatus api={api} />
            <RecentAlerts />
          </aside>
        </div>
      </section>

      <LiveParameters />
      <Analytics />

      <footer className="app-footer">
        <span>AI_SCADA · THERMAL PLANT DIGITAL TWIN</span>
        <span>LIVE VALUES APPEAR WHEN A SUPPORTED DATA SOURCE IS CONNECTED</span>
      </footer>
    </main>
  )
}

export default App
