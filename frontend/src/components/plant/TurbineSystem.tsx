import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import {
  Box,
  Cylinder,
  Foundation,
  InstancedBoxes,
  InstancedCylinders,
  SelectionBounds,
  metal,
  type BoxPlacement,
  type CylinderPlacement,
} from './industrial'
import { VISUAL_ONLY_DRIVE } from './motion'

const bladeAngles = Array.from({ length: 12 }, (_, index) => (index / 12) * Math.PI * 2)
const turbineStages = [
  { x: -2.3, radius: 0.88, length: 2.1 },
  { x: -0.5, radius: 1.1, length: 2.2 },
  { x: 1.45, radius: 1.35, length: 2.55 },
]
const rotorStagePositions = [-2.25, -0.7, 0.8, 2.15]

const rotorBladePlacements: BoxPlacement[][] = rotorStagePositions.map((_, stage) =>
  bladeAngles.map((angle) => ({
    position: [0, Math.cos(angle) * (0.58 + stage * 0.055), Math.sin(angle) * (0.58 + stage * 0.055)],
    size: [0.15, 0.78 + stage * 0.12, 0.09],
    rotation: [angle, 0, 0],
  })),
)

const casingBoltPlacements: CylinderPlacement[] = turbineStages.flatMap(({ x, radius, length }) =>
  [-length / 2, length / 2].flatMap((offset) =>
    Array.from({ length: 12 }, (_, index) => {
      const angle = (index / 12) * Math.PI * 2
      return {
        position: [x + offset, 2.78 + Math.cos(angle) * radius * 0.9, Math.sin(angle) * radius * 0.9],
        radius: 0.045,
        length: 0.13,
        rotation: [0, 0, Math.PI / 2],
      }
    }),
  ),
)

function TurbineStageShell({
  stage,
  position,
  radius,
  length,
  material,
}: {
  stage: number
  position: [number, number, number]
  radius: number
  length: number
  material: THREE.Material
}) {
  const geometry = useMemo(
    () => new THREE.CylinderGeometry(radius, radius, length, 32, 1, false, 0.62, Math.PI * 2 - 1.24),
    [radius, length],
  )

  return (
    <mesh
      name={`TurbineCasingCutaway-${stage}`}
      position={position}
      rotation={[0, 0, Math.PI / 2]}
      geometry={geometry}
      material={material}
      castShadow
    />
  )
}

function TurbineRotor() {
  const rotor = useRef<THREE.Group>(null)
  useFrame((_, delta) => {
    if (rotor.current) rotor.current.rotation.x += delta * (1.6 + VISUAL_ONLY_DRIVE.turbineSpeed * 2.2)
  })

  return (
    <group ref={rotor} name="TurbineRotor" position={[0, 2.85, 0]}>
      <Cylinder
        position={[0, 0, 0]}
        radius={0.27}
        length={9.2}
        material={metal.casingDark}
        rotation={[0, 0, Math.PI / 2]}
        radialSegments={24}
      />
      {rotorStagePositions.map((x, stage) => (
        <group key={x} position={[x, 0, 0]} name={`RotorBladeStage-${stage + 1}`}>
          <InstancedBoxes placements={rotorBladePlacements[stage]} material={metal.casingLight} />
          <Cylinder position={[0, 0, 0]} radius={0.52 + stage * 0.08} length={0.18} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} />
        </group>
      ))}
    </group>
  )
}

function TurbineCasing() {
  return (
    <group name="MultiStageTurbineCasing">
      <Foundation position={[0, 0.12, 0]} size={[9.4, 0.52, 4.8]} />
      <Box position={[0, 0.68, 0]} size={[8.2, 0.55, 3.6]} material={metal.frameDark} />
      {turbineStages.map(({ x, radius, length }, i) => (
        <group key={x} name={`TurbineStage-${i + 1}`}>
          <TurbineStageShell
            stage={i + 1}
            position={[x, 2.78, 0]}
            radius={radius}
            length={length}
            material={i === 2 ? metal.casingLight : metal.casing}
          />
          {[-length / 2, length / 2].map((offset) => (
            <Cylinder key={offset} position={[x + offset, 2.78, 0]} radius={radius * 1.035} length={0.12} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
          ))}
          <Box position={[x, 2.78 + radius * 0.48, 0]} size={[length * 0.84, 0.18, radius * 1.45]} material={metal.casingDark} />
          {[-1, 1].map((side) => (
            <Box key={side} position={[x, 1.78, side * radius * 0.74]} size={[length * 0.85, 0.13, 0.12]} material={metal.frame} />
          ))}
        </group>
      ))}
      <Cylinder position={[-3.63, 2.78, 0]} radius={0.88} length={0.24} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
      <Cylinder position={[2.86, 2.78, 0]} radius={1.3} length={0.24} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
      <InstancedCylinders placements={casingBoltPlacements} material={metal.casingLight} radialSegments={8} />
      <Box position={[-2.32, 3.93, 0]} size={[1.72, 0.52, 1.08]} material={metal.casingDark} />
      <Cylinder position={[-2.32, 4.28, 0]} radius={0.36} length={0.2} material={metal.casingLight} radialSegments={16} />
      <Cylinder position={[-2.32, 4.4, 0]} radius={0.18} length={0.12} material={metal.frameDark} radialSegments={12} />
      {[-3.3, -1.3, 0.65, 2.55].map((x) => (
        <group key={x}>
          <Box position={[x, 1.75, -1.6]} size={[0.42, 2.1, 0.34]} material={metal.casingDark} />
          <Box position={[x, 1.75, 1.6]} size={[0.42, 2.1, 0.34]} material={metal.casingDark} />
          <Box position={[x, 0.72, -1.6]} size={[0.7, 0.14, 0.7]} material={metal.casingLight} />
          <Box position={[x, 0.72, 1.6]} size={[0.7, 0.14, 0.7]} material={metal.casingLight} />
        </group>
      ))}
      {Array.from({ length: 11 }, (_, i) => {
        const x = -3.8 + i * 0.73
        return (
          <Box key={i} position={[x, 3.92, 0]} size={[0.065, 0.14, 2.5]} material={metal.frameDark} />
        )
      })}
    </group>
  )
}

export default function TurbineSystem({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="TurbineSystem"
      position={[6.2, 0, 0]}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[10, 5.5, 5.8]} position={[0, 2.7, 0]} visible={selected} />
      <TurbineCasing />
      <TurbineRotor />
    </group>
  )
}
