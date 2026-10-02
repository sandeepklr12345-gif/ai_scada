import { useMemo } from 'react'
import * as THREE from 'three'
import { Box, InstancedCylinders, Member, SelectionBounds, metal } from './industrial'
import type { CylinderPlacement } from './industrial'
import { RisingVapor, VISUAL_ONLY_DRIVE } from './motion'

const stackRibs: CylinderPlacement[] = Array.from({ length: 12 }, (_, index) => {
  const angle = (index / 12) * Math.PI * 2
  return {
    position: [Math.cos(angle) * 0.715, 4, Math.sin(angle) * 0.715],
    radius: 0.025,
    length: 7.72,
  }
})

export default function Chimney({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  const shellGeometry = useMemo(() => new THREE.CylinderGeometry(0.7, 0.78, 8, 36, 1, true), [])
  const linerGeometry = useMemo(() => new THREE.CylinderGeometry(0.56, 0.64, 7.88, 36, 1, true), [])

  return (
    <group
      name="Chimney"
      position={[-6.4, 13.35, -2.4]}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[3, 9.5, 3]} position={[0, 3.9, 0]} visible={selected} />
      <Box position={[0, 0.18, 0]} size={[2.5, 0.38, 2.5]} material={metal.frameDark} />
      <mesh position={[0, 4, 0]} geometry={shellGeometry} material={metal.casingLight} castShadow receiveShadow />
      <mesh position={[0, 4.02, 0]} geometry={linerGeometry} material={metal.casingDark} />
      <mesh position={[0, 8.05, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.frameDark}>
        <torusGeometry args={[0.73, 0.12, 8, 36]} />
      </mesh>
      <mesh position={[0, 8.09, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.casingLight}>
        <torusGeometry args={[0.65, 0.035, 6, 36]} />
      </mesh>
      <InstancedCylinders placements={stackRibs} material={metal.frame} radialSegments={8} />
      <group name="VisualOnlyStackExhaust" position={[0, 8.33, 0]}>
        <RisingVapor
          count={18}
          height={4.1}
          radius={0.68}
          speed={0.58}
          drive={VISUAL_ONLY_DRIVE.boilerLoad}
          drift={[1.1, 0, 1.3]}
        />
      </group>
      <mesh position={[0, 7.38, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.frame}>
        <torusGeometry args={[0.705, 0.045, 6, 36]} />
      </mesh>
      {[-1, 1].flatMap((x) =>
        [-1, 1].map((z) => (
          <Member
            key={`${x}-${z}`}
            from={[x * 0.56, 0.35, z * 0.56]}
            to={[x * 0.72, 3, z * 0.72]}
            width={0.11}
            material={metal.frameDark}
          />
        )),
      )}
      {[1.3, 2.6, 3.9, 5.2, 6.5].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.frameDark}>
          <torusGeometry args={[0.73 - (y / 8) * 0.035, 0.025, 5, 36]} />
        </mesh>
      ))}
    </group>
  )
}
