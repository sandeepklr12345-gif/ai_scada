import * as THREE from 'three'
import { Box, Cylinder, Foundation, Member, Pipe, SelectionBounds, metal, type Vec3 } from './industrial'
import { FlowingParticles, VISUAL_ONLY_DRIVE } from './motion'

const coalTransferRoute: Vec3[] = [
  [-11.5, 4.02, 0],
  [-10.9, 4.03, 0],
  [-10.2, 4.02, 0],
]

const hopperFunnel = (() => {
  const top = 1.3
  const bottom = 0.36
  const halfDepth = 1.15
  const halfWidth = 1.2
  const outletDepth = 0.28
  const outletWidth = 0.32
  const corners = [
    [-halfWidth, top, -halfDepth],
    [halfWidth, top, -halfDepth],
    [halfWidth, top, halfDepth],
    [-halfWidth, top, halfDepth],
    [-outletWidth, bottom, -outletDepth],
    [outletWidth, bottom, -outletDepth],
    [outletWidth, bottom, outletDepth],
    [-outletWidth, bottom, outletDepth],
  ]
  const positions: number[] = []
  ;[[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]].forEach((face) => {
    const [a, b, c, d] = face.map((i) => corners[i])
    positions.push(...a, ...b, ...d, ...b, ...c, ...d)
  })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.computeVertexNormals()
  return geometry
})()

function CoalHopper({ position, index }: { position: Vec3; index: number }) {
  return (
    <group position={position} name={`CoalHopper-${index}`}>
      <Foundation position={[0, -1.55, 0]} size={[3, 0.4, 2.8]} />
      <mesh position={[0, 0, 0]} geometry={hopperFunnel} material={metal.casing} castShadow />
      <Box position={[0, 1.34, 0]} size={[2.66, 0.18, 2.46]} material={metal.frameDark} />
      <Box position={[0, 1.24, 0]} size={[2.28, 0.22, 2.08]} material={metal.coal} />
      <Box position={[0, 0.04, 0]} size={[0.66, 0.38, 0.6]} material={metal.frameDark} />
      {[-1, 1].flatMap((x) =>
        [-1, 1].map((z) => (
          <Member
            key={`${x}-${z}`}
            from={[x * 1.05, 0.04, z * 1.05]}
            to={[x * 0.93, -1.45, z * 0.93]}
            width={0.1}
            material={metal.frame}
          />
        )),
      )}
      <Box position={[0, -1.42, 0]} size={[2.2, 0.12, 2.2]} material={metal.frameDark} />
      <Cylinder position={[0, -0.05, 0]} radius={0.22} length={1.1} material={metal.belt} />
    </group>
  )
}

function CoalMill({
  position,
  index,
  selected,
  onSelect,
}: {
  position: Vec3
  index: number
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      position={position}
      name={`CoalMill-${index}`}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[3.2, 3.5, 3.1]} position={[0, 1.4, 0]} visible={selected} />
      <Foundation position={[0, -0.25, 0]} size={[2.9, 0.5, 2.7]} />
      <Box position={[0, 0.42, 0]} size={[2.45, 0.22, 2.3]} material={metal.frameDark} />
      <Cylinder position={[0, 1.2, 0]} radius={0.8} length={1.55} material={metal.casing} />
      <Cylinder position={[0, 2.03, 0]} radius={0.62} length={0.26} material={metal.casingDark} />
      <Cylinder position={[0, 2.26, 0]} radius={0.5} length={0.24} material={metal.casingLight} />
      <Box position={[1.05, 0.95, 0]} size={[0.95, 0.88, 0.9]} material={metal.casingDark} />
      <Cylinder
        position={[1.45, 0.97, 0]}
        radius={0.31}
        length={0.26}
        material={metal.casingLight}
        rotation={[0, 0, Math.PI / 2]}
      />
      <Cylinder
        position={[0, 2.68, 0]}
        radius={0.23}
        length={0.7}
        material={metal.frameDark}
      />
      {[-0.75, 0, 0.75].map((z) => (
        <Box
          key={z}
          position={[0, 1.2, z]}
          size={[2.48, 0.09, 0.07]}
          material={metal.frameDark}
        />
      ))}
    </group>
  )
}

function BeltConveyor({
  position,
  length,
  slope = 0,
  turn = 0,
}: {
  position: Vec3
  length: number
  slope?: number
  turn?: number
}) {
  const legY = 0.2 - position[1]
  return (
    <group position={position} rotation={[0, turn, slope]} name="CoalTransferBelt">
      <Box position={[0, 0, 0]} size={[length, 0.32, 1.16]} material={metal.frameDark} />
      <Box position={[0, 0.22, 0]} size={[length * 0.94, 0.12, 0.84]} material={metal.belt} />
      {[-0.49, 0.49].map((z) => (
        <Box key={z} position={[0, 0.28, z]} size={[length * 0.94, 0.16, 0.08]} material={metal.frame} />
      ))}
      {Array.from({ length: Math.floor(length / 1.1) }, (_, i) => {
        const x = -length / 2 + 0.55 + i * 1.1
        return (
          <group key={x}>
            <Cylinder position={[x, -0.02, 0]} radius={0.13} length={0.96} rotation={[Math.PI / 2, 0, 0]} material={metal.frame} />
            <Member from={[x, legY, -0.48]} to={[x, -0.16, -0.48]} width={0.08} />
            <Member from={[x, legY, 0.48]} to={[x, -0.16, 0.48]} width={0.08} />
          </group>
        )
      })}
    </group>
  )
}

export default function CoalHandlingSystem({
  selected,
  onSelect,
  selectedMillIndex,
  onSelectMill,
}: {
  selected: boolean
  onSelect: () => void
  selectedMillIndex: number | null
  onSelectMill: (index: number) => void
}) {
  const hoppers: Vec3[] = [
    [-14.2, 1.65, -3.1],
    [-14.2, 1.65, 0],
    [-14.2, 1.65, 3.1],
  ]
  const mills: Vec3[] = [
    [-10, 0.55, -2.65],
    [-10, 0.55, 0],
    [-10, 0.55, 2.65],
  ]

  return (
    <group
      name="CoalHandlingSystem"
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[8.3, 6, 9]} position={[-11.3, 1.3, 0]} visible={selected} />
      <group name="CoalStorageHoppers">
        {hoppers.map((position, i) => (
          <CoalHopper key={i} position={position} index={i + 1} />
        ))}
      </group>
      <group name="CoalConveyors">
        <BeltConveyor position={[-11.8, 1.45, 0]} length={7} turn={Math.PI / 2} />
        <BeltConveyor position={[-9.5, 3.8, 0]} length={4.2} slope={0.04} />
        {hoppers.map(([x, , z], i) => (
          <Pipe
            key={i}
            points={[[x, 1.24, z], [x + 0.45, 1.45, z], [-11.8, 1.58, z]]}
            radius={0.17}
            material={metal.coal}
            segments={16}
          />
        ))}
        <Pipe
          points={[[-11.8, 1.6, 0], [-11.35, 2.15, 0], [-10.75, 2.85, 0], [-10.05, 3.55, 0]]}
          radius={0.28}
          material={metal.coal}
          segments={18}
        />
        <FlowingParticles points={coalTransferRoute} color="#242729" count={8} radius={0.085} speed={0.055} drive={VISUAL_ONLY_DRIVE.coalFlow} offset={[0, 0.15, 0.12]} faceted />
        {[-0.68, 0, 0.68].map((z) => (
          <Pipe
            key={z}
            points={[[-10, 3.7, 0], [-10, 3.45, z], [-10, 3.28, z * 3.9]]}
            radius={0.12}
            material={metal.coal}
            segments={18}
          />
        ))}
        {[-0.68, 0, 0.68].map((z) => (
          <FlowingParticles
            key={`coal-flow-${z}`}
            points={[[-10, 3.68, 0], [-10, 3.45, z], [-10, 3.28, z * 3.9]]}
            color="#242729"
            count={4}
            radius={0.07}
            speed={0.045}
            drive={VISUAL_ONLY_DRIVE.coalFlow}
            offset={[0, 0.13, 0]}
            faceted
          />
        ))}
      </group>
      <group name="CoalMills">
        {mills.map((position, i) => (
          <CoalMill
            key={i}
            position={position}
            index={i + 1}
            selected={selectedMillIndex === i + 1}
            onSelect={() => onSelectMill(i + 1)}
          />
        ))}
      </group>
      <group name="PulverizedCoalBurnerLines">
        {mills.map(([, , z], i) => {
          const y = 5 + i * 2.3
          const route: Vec3[] = [
            [-10, 3.15, z],
            [-9.15, 3.65, z],
            [-8.55, y, z],
            [-7.3, y, z],
            [-5.55, y, z],
          ]
          return (
            <group key={z}>
              <Pipe points={route} radius={0.15} material={metal.coal} segments={26} />
              <FlowingParticles points={route} color="#343638" count={5} radius={0.066} speed={0.05} drive={VISUAL_ONLY_DRIVE.coalFlow} offset={[0, 0.15, 0.13]} faceted />
            </group>
          )
        })}
      </group>
      <Box position={[-9.65, 0.08, 0]} size={[6.1, 0.16, 9.2]} material={metal.concrete} />
      <Box position={[-8.25, 4.7, 0]} size={[0.3, 0.3, 8]} material={metal.frameDark} />
    </group>
  )
}
