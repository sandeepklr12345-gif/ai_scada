import { Box, Cylinder, Foundation, SelectionBounds, metal } from './industrial'
import { RotatingPumpCoupling, VISUAL_ONLY_DRIVE } from './motion'

function FeedwaterPump({
  position,
  index,
  selected,
  onSelect,
}: {
  position: [number, number, number]
  index: number
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      position={position}
      name={`FeedwaterPump-${index}`}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[2.5, 2.3, 2.4]} position={[0, 0.95, 0]} visible={selected} />
      <Foundation position={[0, 0.15, 0]} size={[2.25, 0.38, 2.15]} />
      <Box position={[-0.38, 0.72, 0]} size={[1.08, 0.92, 1.06]} material={metal.casingDark} />
      <Cylinder position={[-0.38, 0.72, 0]} radius={0.46} length={0.26} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[0.53, 0.72, 0]} radius={0.39} length={1.1} material={metal.casing} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[1.08, 0.72, 0]} radius={0.3} length={0.16} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[-0.7, 0.72, 0]} radius={0.29} length={0.14} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} />
      <RotatingPumpCoupling drive={VISUAL_ONLY_DRIVE.feedwaterFlow} />
      {[-0.72, 0.72].map((x) => (
        <Box key={x} position={[x, 1.24, 0]} size={[0.13, 0.16, 0.8]} material={metal.casingLight} />
      ))}
      <Box position={[0.92, 0.22, 0]} size={[0.26, 0.5, 0.32]} material={metal.frameDark} />
    </group>
  )
}

export default function FeedwaterSystem({
  selected,
  onSelect,
  selectedPumpIndex,
  onSelectPump,
}: {
  selected: boolean
  onSelect: () => void
  selectedPumpIndex: number | null
  onSelectPump: (index: number) => void
}) {
  return (
    <group
      name="FeedwaterSystem"
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[6.8, 3.5, 4.1]} position={[0.25, 1.75, 8.2]} visible={selected} />
      <group name="FeedwaterPumps">
        {[-1.9, 0.25, 2.4].map((x, i) => (
          <FeedwaterPump
            key={x}
            position={[x, 0, 8.2]}
            index={i + 1}
            selected={selectedPumpIndex === i + 1}
            onSelect={() => onSelectPump(i + 1)}
          />
        ))}
      </group>
      <group name="FeedwaterManifolds">
        <Box position={[-0.45, 0.96, 8.2]} size={[5.8, 0.24, 0.34]} material={metal.feedwater} />
        <Box position={[1.3, 1.05, 8.45]} size={[5.7, 0.24, 0.34]} material={metal.feedwater} />
      </group>
      <Box position={[0.2, 0.12, 8.2]} size={[7.3, 0.25, 4.4]} material={metal.concrete} />
    </group>
  )
}
