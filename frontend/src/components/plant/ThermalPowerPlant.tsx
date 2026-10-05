import { useEffect, useRef } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { Environment, Grid, Html, OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import CoalHandlingSystem from './CoalSystem'
import BoilerSystem from './BoilerSystem'
import Chimney from './Chimney'
import CondenserSystem from './CondenserSystem'
import CoolingSystem from './CoolingSystem'
import FeedwaterSystem from './FeedwaterSystem'
import Generator from './Generator'
import ProcessPipes from './ProcessPipes'
import TurbineSystem from './TurbineSystem'
import type { EquipmentId, EquipmentSelection } from './equipment'
import { Box, metal } from './industrial'

const plantLabels = [
  { text: 'COAL', position: [-14.2, 4.35, 0] as [number, number, number] },
  { text: 'COAL MILL', position: [-10, 4.55, 0] as [number, number, number] },
  { text: 'BOILER', position: [-3.5, 17.35, 0] as [number, number, number] },
  { text: 'TURBINE', position: [6.2, 5.15, 0] as [number, number, number] },
  { text: 'CONDENSER', position: [4.8, 3.7, 3.9] as [number, number, number] },
  { text: 'FEEDWATER PUMP', position: [0.25, 3.25, 8.2] as [number, number, number] },
  { text: 'GENERATOR', position: [12.8, 5.1, 0] as [number, number, number] },
  { text: 'COOLING SYSTEM', position: [11.2, 15.7, -9.5] as [number, number, number] },
]

function PlantLabels() {
  return (
    <group name="PlantSystemLabels">
      {plantLabels.map(({ text, position }) => (
        <Html key={text} position={position} center pointerEvents="none" zIndexRange={[3, 0]}>
          <div
            style={{
              padding: '4px 7px',
              border: '1px solid rgba(195, 219, 228, 0.62)',
              borderRadius: 3,
              color: '#f0f4f4',
              background: 'rgba(13, 24, 30, 0.86)',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.28)',
              font: '600 9px/1.1 Inter, Arial, sans-serif',
              letterSpacing: '0.08em',
              whiteSpace: 'nowrap',
              userSelect: 'none',
            }}
          >
            {text}
          </div>
        </Html>
      ))}
    </group>
  )
}

function PlantControls() {
  const controls = useRef<OrbitControlsImpl>(null)
  const { gl } = useThree()

  useEffect(() => {
    const handleWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey) return
      event.preventDefault()
      const factor = Math.pow(1.12, Math.min(3, Math.abs(event.deltaY) / 120))
      if (event.deltaY > 0) controls.current?.dollyOut(factor)
      else if (event.deltaY < 0) controls.current?.dollyIn(factor)
    }

    const canvas = gl.domElement
    canvas.addEventListener('wheel', handleWheel, { passive: false })
    return () => canvas.removeEventListener('wheel', handleWheel)
  }, [gl])

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      target={[0, 8.4, -0.8]}
      enableDamping
      dampingFactor={0.08}
      enableRotate
      enablePan
      enableZoom={false}
      minDistance={24}
      maxDistance={78}
      minPolarAngle={0.18}
      maxPolarAngle={Math.PI / 2 - 0.035}
    />
  )
}

function PlantModel({
  selected,
  setSelected,
}: {
  selected: EquipmentSelection | null
  setSelected: (equipment: EquipmentSelection | null) => void
}) {
  const select = (id: EquipmentId, instance?: number) => () => setSelected({ id, instance })

  return (
    <group
      name="ThermalPowerPlant"
      onClick={(event) => {
        event.stopPropagation()
        setSelected(null)
      }}
    >
      <Box position={[0, -0.08, 0]} size={[36, 0.36, 30]} material={metal.concrete} receiveShadow />
      <Box position={[0, 0.12, 0]} size={[35.5, 0.08, 29.5]} material={metal.frameDark} />
      <CoalHandlingSystem
        selected={selected?.id === 'CoalHandlingSystem'}
        onSelect={select('CoalHandlingSystem')}
        selectedMillIndex={selected?.id === 'CoalMill' ? selected.instance ?? null : null}
        onSelectMill={(instance) => setSelected({ id: 'CoalMill', instance })}
      />
      <BoilerSystem selected={selected?.id === 'BoilerSystem'} onSelect={select('BoilerSystem')} />
      <Chimney selected={selected?.id === 'Chimney'} onSelect={select('Chimney')} />
      <TurbineSystem selected={selected?.id === 'TurbineSystem'} onSelect={select('TurbineSystem')} />
      <Generator selected={selected?.id === 'Generator'} onSelect={select('Generator')} />
      <CondenserSystem selected={selected?.id === 'CondenserSystem'} onSelect={select('CondenserSystem')} />
      <FeedwaterSystem
        selected={selected?.id === 'FeedwaterSystem'}
        onSelect={select('FeedwaterSystem')}
        selectedPumpIndex={selected?.id === 'FeedwaterPump' ? selected.instance ?? null : null}
        onSelectPump={(instance) => setSelected({ id: 'FeedwaterPump', instance })}
      />
      <CoolingSystem selected={selected?.id === 'CoolingSystem'} onSelect={select('CoolingSystem')} />
      <ProcessPipes selected={selected?.id === 'ProcessPipes'} onSelect={select('ProcessPipes')} />
    </group>
  )
}

function ThermalPowerPlant({
  selection,
  onSelectionChange,
}: {
  selection: EquipmentSelection | null
  onSelectionChange: (selection: EquipmentSelection | null) => void
}) {

  return (
    <Canvas
      camera={{
        position: [27, 26, 30],
        fov: 34,
        near: 0.1,
        far: 180,
      }}
      dpr={[1, 1.5]}
      shadows
      gl={{ antialias: true, alpha: false }}
      onPointerMissed={() => onSelectionChange(null)}
    >
      <color attach="background" args={['#071018']} />
      <ambientLight intensity={0.72} />
      <directionalLight
        position={[16, 24, 18]}
        intensity={3.1}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-24}
        shadow-camera-right={24}
        shadow-camera-top={26}
        shadow-camera-bottom={-24}
        shadow-bias={-0.0003}
      />
      <directionalLight position={[-16, 12, -10]} intensity={0.85} />
      <Environment preset="studio" />
      <mesh position={[0, -0.265, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[58, 52]} />
        <shadowMaterial transparent opacity={0.28} />
      </mesh>
      <Grid
        args={[46, 42]}
        position={[0, 0.14, 0]}
        cellSize={1}
        cellThickness={0.35}
        cellColor="#20313a"
        sectionSize={5}
        sectionThickness={0.65}
        sectionColor="#344650"
        fadeDistance={48}
        fadeStrength={1}
        infiniteGrid
      />
      <PlantModel selected={selection} setSelected={onSelectionChange} />
      <PlantLabels />
      <PlantControls />
    </Canvas>
  )
}

export default ThermalPowerPlant
