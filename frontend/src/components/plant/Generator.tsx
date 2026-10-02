import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Box, Cylinder, Foundation, InstancedBoxes, SelectionBounds, metal } from './industrial'
import type { BoxPlacement } from './industrial'
import { VISUAL_ONLY_DRIVE } from './motion'

const generatorRibs: BoxPlacement[] = Array.from({ length: 12 }, (_, i) => ({
  position: [0, 3, -1.47 + i * 0.267],
  size: [4.05, 1.9, 0.075],
}))

function GeneratorCoupling() {
  const coupling = useRef<THREE.Group>(null)
  useFrame((_, delta) => {
    if (coupling.current) coupling.current.rotation.x += delta * (1.6 + VISUAL_ONLY_DRIVE.turbineSpeed * 2.2)
  })

  return (
    <group ref={coupling} position={[-3.05, 2.85, 0]} name="GeneratorShaftCoupling">
      <Cylinder position={[0, 0, 0]} radius={0.24} length={1.38} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} radialSegments={20} />
      {[-0.55, 0.55].map((x) => (
        <group key={x} position={[x, 0, 0]}>
          <Cylinder position={[0, 0, 0]} radius={0.37} length={0.18} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} radialSegments={24} />
          {Array.from({ length: 8 }, (_, index) => {
            const angle = (index / 8) * Math.PI * 2
            return (
              <Cylinder
                key={index}
                position={[0, Math.cos(angle) * 0.33, Math.sin(angle) * 0.33]}
                radius={0.04}
                length={0.2}
                material={metal.casingLight}
                rotation={[0, 0, Math.PI / 2]}
                radialSegments={8}
              />
            )
          })}
        </group>
      ))}
    </group>
  )
}

export default function Generator({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="Generator"
      position={[12.8, 0, 0]}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[6.2, 4.7, 4.9]} position={[0, 2.45, 0]} visible={selected} />
      <Foundation position={[0, 0.16, 0]} size={[6.6, 0.48, 5.1]} />
      <Box position={[0, 2.85, 0]} size={[5.2, 2.85, 3.5]} material={metal.generator} castShadow receiveShadow />
      <Cylinder position={[0, 2.85, 0]} radius={1.42} length={4.8} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
      <Cylinder position={[-2.45, 2.85, 0]} radius={1.42} length={0.16} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
      <Cylinder position={[2.45, 2.85, 0]} radius={1.42} length={0.16} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} radialSegments={32} />
      <GeneratorCoupling />
      <Box position={[0, 4.33, 0]} size={[4.5, 0.16, 3.1]} material={metal.generator} />
      <InstancedBoxes placements={generatorRibs} material={metal.casingDark} />
      {[-1.35, 1.35].flatMap((z) =>
        [-1.9, 1.9].map((x) => (
          <group key={`${x}-${z}`}>
            <Box position={[x, 1, z]} size={[0.48, 0.85, 0.48]} material={metal.frameDark} />
            <Box position={[x, 0.42, z]} size={[0.8, 0.12, 0.8]} material={metal.casingLight} />
          </group>
        )),
      )}
      <Box position={[-2.73, 1.42, 0]} size={[0.35, 0.9, 0.85]} material={metal.casingDark} />
      <Cylinder position={[-2.91, 2.85, 0]} radius={0.32} length={0.24} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} />
      <Box position={[0.3, 4.49, 0]} size={[1.8, 0.2, 1.1]} material={metal.casingDark} />
      <Box position={[1.42, 4.43, 0]} size={[0.36, 0.25, 1.7]} material={metal.casingLight} />
    </group>
  )
}
