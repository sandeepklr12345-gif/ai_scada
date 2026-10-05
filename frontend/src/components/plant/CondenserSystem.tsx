import { BoltedFlange, Box, Cylinder, Foundation, InstancedBoxes, InstancedCylinders, SelectionBounds, metal } from './industrial'
import type { BoxPlacement, CylinderPlacement } from './industrial'

const condenserFins: BoxPlacement[] = Array.from({ length: 17 }, (_, index) => ({
  position: [-2.7 + index * 0.337, 2.32, 0],
  size: [0.045, 0.22, 3.7],
}))

const condenserTubeEnds: CylinderPlacement[] = [-1, 1].flatMap((side) =>
  Array.from({ length: 60 }, (_, index) => {
    const column = index % 15
    const row = Math.floor(index / 15)
    return {
      position: [-2.8 + column * 0.4, 0.9 + row * 0.32, side * 2.25],
      radius: 0.045,
      length: 0.1,
      rotation: [Math.PI / 2, 0, 0],
    }
  }),
)

export default function CondenserSystem({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="CondenserSystem"
      position={[4.8, 0, 3.9]}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[7.6, 3.6, 5.2]} position={[0, 1.8, 0]} visible={selected} />
      <Foundation position={[0, 0.2, 0]} size={[8.2, 0.55, 5.8]} />
      <Box position={[0, 1.42, 0]} size={[6.8, 1.7, 4.4]} material={metal.casingDark} castShadow receiveShadow />
      <Cylinder position={[-0.1, 1.55, -0.78]} radius={0.78} length={6.15} material={metal.casing} rotation={[0, 0, Math.PI / 2]} radialSegments={28} />
      <Cylinder position={[-0.1, 1.55, 0.78]} radius={0.78} length={6.15} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} radialSegments={28} />
      {[-0.78, 0.78].flatMap((z) => [-3.05, 2.85].map((x) => (
        <BoltedFlange key={`${x}-${z}`} position={[x, 1.55, z]} radius={0.86} axis="x" thickness={0.12} />
      )))}
      {[-0.78, 0.78].flatMap((z) => [-2.35, -0.8, 0.75, 2.3].map((x) => (
        <mesh key={`${x}-${z}-joint`} position={[x, 1.55, z]} rotation={[0, Math.PI / 2, 0]} material={metal.frame}>
          <torusGeometry args={[0.79, 0.025, 6, 28]} />
        </mesh>
      )))}
      <InstancedCylinders placements={condenserTubeEnds} material={metal.casingLight} radialSegments={10} />
      {[-3.2, 3.2].map((x) => (
        <group key={x}>
          <Box position={[x, 0.86, 0]} size={[0.34, 1.25, 4.65]} material={metal.frameDark} />
          <Box position={[x, 0.27, 0]} size={[0.72, 0.15, 5]} material={metal.casingLight} />
          <Cylinder position={[x, 1.55, 0]} radius={0.9} length={0.14} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} radialSegments={28} />
        </group>
      ))}
      <InstancedBoxes placements={condenserFins} material={metal.frame} />
      <Box position={[0, 2.38, 0]} size={[1.35, 0.24, 1.35]} material={metal.casingDark} />
      <Cylinder position={[0, 2.7, 0]} radius={0.46} length={0.55} material={metal.casingLight} />
      <Cylinder position={[0, 2.95, 0]} radius={0.19} length={0.62} material={metal.casingDark} />
      <Box position={[0, 3.34, 0]} size={[1.15, 0.12, 1.05]} material={metal.frameDark} />
    </group>
  )
}
