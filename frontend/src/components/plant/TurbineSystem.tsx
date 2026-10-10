import { useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import {
  Box,
  Cylinder,
  Foundation,
  InstancedCylinders,
  SelectionBounds,
  metal,
  type CylinderPlacement,
} from './industrial'
import { VISUAL_ONLY_DRIVE } from './motion'

const turbineStages = [
  { x: -2.3, radius: 0.88, length: 2.1 },
  { x: -0.5, radius: 1.1, length: 2.2 },
  { x: 1.45, radius: 1.35, length: 2.55 },
]
const rotorStagePositions = [-2.25, -0.7, 0.8, 2.15]
const rotorBladeRows = [
  { outerRadius: 0.66, chord: 0.48, count: 18 },
  { outerRadius: 0.78, chord: 0.43, count: 20 },
  { outerRadius: 0.94, chord: 0.38, count: 22 },
  { outerRadius: 1.1, chord: 0.34, count: 24 },
]

function createTurbineBladeGeometry(outerRadius: number, chord: number) {
  const geometry = new THREE.BufferGeometry()
  const radialSegments = 6
  const chordSegments = 3
  const rootRadius = 0.31
  const bladeSpan = outerRadius - rootRadius
  const positions: number[] = []
  const indices: number[] = []
  const surfaceVertexCount = (radialSegments + 1) * (chordSegments + 1)

  for (const side of [-1, 1]) {
    const vertexOffset = positions.length / 3
    for (let radial = 0; radial <= radialSegments; radial += 1) {
      const t = radial / radialSegments
      const radius = rootRadius + bladeSpan * t
      const localChord = chord * (0.72 + 0.28 * Math.sin(Math.PI * t))
      const sweep = 0.08 * t
      const thickness = 0.018 * (1 - 0.45 * t)

      for (let section = 0; section <= chordSegments; section += 1) {
        const u = section / chordSegments
        const chordOffset = u - 0.5
        const camber = Math.sin(u * Math.PI) * 0.065 * (1 - 0.3 * t)
        positions.push(
          sweep + chordOffset * localChord,
          radius,
          camber + side * thickness,
        )
      }
    }

    for (let radial = 0; radial < radialSegments; radial += 1) {
      for (let section = 0; section < chordSegments; section += 1) {
        const a = vertexOffset + radial * (chordSegments + 1) + section
        const b = a + chordSegments + 1
        if (side > 0) indices.push(a, b, a + 1, b, b + 1, a + 1)
        else indices.push(a, a + 1, b, b, a + 1, b + 1)
      }
    }
  }

  // Close the root, tip, leading edge, and trailing edge so each blade has a
  // real, shaded thickness when the rotor turns through the cutaway casing.
  const backOffset = surfaceVertexCount
  const addBridge = (frontIndex: number, backIndex: number, nextFront: number, nextBack: number) => {
    indices.push(frontIndex, nextFront, backIndex, nextFront, nextBack, backIndex)
  }
  for (let section = 0; section < chordSegments; section += 1) {
    const frontRoot = section
    const backRoot = backOffset + section
    addBridge(frontRoot, backRoot, frontRoot + 1, backRoot + 1)
    const frontTip = radialSegments * (chordSegments + 1) + section
    const backTip = backOffset + frontTip
    addBridge(frontTip + 1, backTip + 1, frontTip, backTip)
  }
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const frontLeading = radial * (chordSegments + 1)
    const backLeading = backOffset + frontLeading
    addBridge(frontLeading + chordSegments + 1, backLeading + chordSegments + 1, frontLeading, backLeading)
    const frontTrailing = frontLeading + chordSegments
    const backTrailing = backLeading + chordSegments
    addBridge(frontTrailing, backTrailing, frontTrailing + chordSegments + 1, backTrailing + chordSegments + 1)
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

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
      receiveShadow
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
          <AxialRotorBladeRow {...rotorBladeRows[stage]} />
          <Cylinder position={[0, 0, 0]} radius={0.43 + stage * 0.035} length={0.2} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
          <Cylinder position={[0.12, 0, 0]} radius={0.36 + stage * 0.03} length={0.055} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
        </group>
      ))}
    </group>
  )
}

function AxialRotorBladeRow({ outerRadius, chord, count }: (typeof rotorBladeRows)[number]) {
  const instances = useRef<THREE.InstancedMesh>(null)
  const geometry = useMemo(() => createTurbineBladeGeometry(outerRadius, chord), [outerRadius, chord])

  useLayoutEffect(() => {
    const mesh = instances.current
    if (!mesh) return
    const transform = new THREE.Object3D()
    for (let index = 0; index < count; index += 1) {
      transform.rotation.set((index / count) * Math.PI * 2, 0, 0)
      transform.updateMatrix()
      mesh.setMatrixAt(index, transform.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [count])

  return (
    <instancedMesh
      ref={instances}
      args={[geometry, undefined, count]}
      castShadow
      receiveShadow
      name="CurvedAxialTurbineBlades"
    >
      <meshStandardMaterial
        color="#bac3c2"
        metalness={0.82}
        roughness={0.31}
        side={THREE.DoubleSide}
      />
    </instancedMesh>
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
