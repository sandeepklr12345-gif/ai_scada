import { BoltedFlange, Box, Cylinder, Foundation, InstancedBoxes, InstancedCylinders, Member, SelectionBounds, metal } from './industrial'
import type { BoxPlacement, CylinderPlacement } from './industrial'
import { FireboxGlow, VISUAL_ONLY_DRIVE } from './motion'

const levels = [0.65, 3.65, 6.65, 9.65, 12.65, 15.65]
const frameCorners = [-4.85, 4.85] as const
const tubeBankBright: CylinderPlacement[] = []
const tubeBankMain: CylinderPlacement[] = []
for (const x of [-2.5, 2.5]) {
  for (let i = 0; i < 18; i += 1) {
    const placement = { position: [x, 7, -2.3 + i * 0.27] as [number, number, number], radius: 0.055, length: 7.5 }
    ;(i % 3 === 0 ? tubeBankBright : tubeBankMain).push(placement)
  }
}
const superheaterHeaders: CylinderPlacement[] = [-2.35, -1.55, -0.75, 0.05, 0.85, 1.65, 2.45].map((x) => ({
  position: [x, 10.6, 2.38], radius: 0.06, length: 3.8, rotation: [Math.PI / 2, 0, 0],
}))

// Keep the pressure-wall details in a few instance batches so the boiler can
// show its tube construction without adding dozens of individual draw calls.
const waterwallTubes: CylinderPlacement[] = []
for (const x of [-2.85, -2.48, -2.11, -1.78, 1.78, 2.11, 2.48, 2.85]) {
  waterwallTubes.push({ position: [x, 5.2, 2.72], radius: 0.045, length: 8.35 })
}
for (const side of [-1, 1]) {
  for (let index = 0; index < 15; index += 1) {
    const z = -2.45 + index * 0.35
    waterwallTubes.push({ position: [side * 3.075, 5.2, z], radius: 0.045, length: 8.35 })
  }
}

const boilerDowncomers: CylinderPlacement[] = []
for (const x of [-3.2, 3.2]) {
  for (const z of [-2.25, 2.25]) {
    boilerDowncomers.push({ position: [x, 5.2, z], radius: 0.105, length: 9.05 })
  }
}

const waterwallHeaders: CylinderPlacement[] = []
for (const z of [-2.86, 2.86]) {
  for (const y of [1.0, 9.35]) {
    waterwallHeaders.push({ position: [0, y, z], radius: 0.14, length: 6.15, rotation: [0, 0, Math.PI / 2] })
  }
}
for (const x of [-3.18, 3.18]) {
  for (const y of [1.0, 9.35]) {
    waterwallHeaders.push({ position: [x, y, 0], radius: 0.14, length: 5.2, rotation: [Math.PI / 2, 0, 0] })
  }
}

const burnerNozzleCollars: CylinderPlacement[] = [-1.12, 0, 1.12].map((x) => ({
  position: [x, 4.18, 3.015], radius: 0.275, length: 0.13, rotation: [Math.PI / 2, 0, 0],
}))
const burnerNozzles: CylinderPlacement[] = [-1.12, 0, 1.12].map((x) => ({
  position: [x, 4.18, 3.2], radius: 0.145, length: 0.28, rotation: [Math.PI / 2, 0, 0],
}))

const burnerDoorFrames: BoxPlacement[] = []
const burnerGrating: BoxPlacement[] = []
for (const x of [-1.12, 0, 1.12]) {
  burnerDoorFrames.push(
    { position: [x - 0.34, 5.1, 2.81], size: [0.14, 1, 0.1] },
    { position: [x + 0.34, 5.1, 2.81], size: [0.14, 1, 0.1] },
    { position: [x, 4.58, 2.81], size: [0.82, 0.16, 0.1] },
    { position: [x, 5.62, 2.81], size: [0.82, 0.16, 0.1] },
  )
  for (const offset of [-0.27, -0.09, 0.09, 0.27]) {
    burnerGrating.push({ position: [x, 5.1 + offset, 2.82], size: [0.56, 0.025, 0.035] })
  }
  for (const offset of [-0.18, 0.18]) {
    burnerGrating.push({ position: [x + offset, 5.1, 2.82], size: [0.025, 0.82, 0.035] })
  }
}

const sideGuardRails: BoxPlacement[] = []
for (const y of [2.5, 5.5, 8.5, 11.5, 14.5]) {
  for (const x of [-4.56, 4.56]) {
    for (const height of [0.58, 1.08]) sideGuardRails.push({ position: [x, y + height, 0], size: [0.075, 0.07, 5.8] })
    for (const z of [-2.75, -1.4, 0, 1.4, 2.75]) sideGuardRails.push({ position: [x, y + 0.56, z], size: [0.07, 1.08, 0.07] })
  }
}

const ladderDetails: BoxPlacement[] = []
for (const bottom of [2.5, 5.5, 8.5, 11.5]) {
  for (const z of [-1.1, -0.55]) ladderDetails.push({ position: [-4.63, bottom + 1.42, z], size: [0.065, 2.82, 0.065] })
  for (let rung = 0; rung < 8; rung += 1) {
    ladderDetails.push({ position: [-4.63, bottom + 0.28 + rung * 0.32, -0.825], size: [0.065, 0.05, 0.55] })
  }
}

function BoilerFrame() {
  return (
    <group name="BoilerStructuralFrame">
      {frameCorners.flatMap((x) =>
        frameCorners.map((z) => (
          <group key={`${x}-${z}`}>
            <Member from={[x, 0.42, z]} to={[x, 16.2, z]} width={0.24} depth={0.24} material={metal.furnaceDoor} />
            {levels.slice(1).map((y) => (
              <Member
                key={y}
                from={[x - 0.27, y, z]}
                to={[x + 0.27, y, z]}
                width={0.15}
                depth={0.24}
                material={metal.boilerFrameDark}
              />
            ))}
          </group>
        )),
      )}
      {levels.map((y) => (
        <group key={y}>
          {[-3.5, 3.5].map((z) => (
            <Box key={z} position={[0, y, z]} size={[9.7, 0.18, 0.18]} material={metal.furnaceDoor} />
          ))}
          {[-4.7, 4.7].map((x) => (
            <Box key={x} position={[x, y, 0]} size={[0.18, 0.18, 7.1]} material={metal.furnaceDoor} />
          ))}
        </group>
      ))}
      {[-3.5, 3.5].flatMap((z) =>
        [0.65, 3.65, 6.65, 9.65, 12.65].map((bottom) => (
          <Member
            key={`${z}-${bottom}`}
            from={[-4.7, bottom, z]}
            to={[4.7, bottom + 3, z]}
            width={0.11}
            material={metal.boilerFrameDark}
          />
        )),
      )}
      {[-4.7, 4.7].flatMap((x) =>
        [0.65, 3.65, 6.65, 9.65, 12.65].map((bottom) => (
          <Member
            key={`${x}-${bottom}`}
            from={[x, bottom, -3.4]}
            to={[x, bottom + 3, 3.4]}
            width={0.11}
            material={metal.boilerFrameDark}
          />
        )),
      )}
      {[2.5, 5.5, 8.5, 11.5, 14.5].map((y) => (
        <group key={y}>
          <Box position={[0, y, -3.25]} size={[9.5, 0.12, 0.55]} material={metal.platform} />
          <Box position={[0, y, 3.25]} size={[9.5, 0.12, 0.55]} material={metal.platform} />
          <Box position={[-4.45, y, 0]} size={[0.65, 0.12, 5.8]} material={metal.platform} />
          <Box position={[4.45, y, 0]} size={[0.65, 0.12, 5.8]} material={metal.platform} />
          {[-4.5, 0, 4.5].map((x) => (
            <Box key={x} position={[x, y + 0.58, 3.42]} size={[0.08, 1.1, 0.08]} material={metal.rail} />
          ))}
          <Box position={[0, y + 1.08, 3.42]} size={[9.1, 0.08, 0.08]} material={metal.rail} />
          <Box position={[0, y + 0.58, 3.42]} size={[9.1, 0.06, 0.06]} material={metal.rail} />
        </group>
      ))}
      <InstancedBoxes placements={sideGuardRails} material={metal.rail} />
      <InstancedBoxes placements={ladderDetails} material={metal.frameDark} />
    </group>
  )
}

function BoilerFurnace() {
  return (
    <group name="BoilerFurnace">
      <Foundation position={[0, 0.05, 0]} size={[10.5, 0.75, 8.2]} />
      <Box position={[0, 5.2, 0]} size={[6.1, 9.1, 5.3]} material={metal.furnace} castShadow receiveShadow />
      <Box position={[0, 10.6, 0]} size={[4.65, 2.5, 4.6]} material={metal.casingDark} castShadow receiveShadow />
      <Box position={[0, 12.95, 0]} size={[5.1, 1.2, 5]} material={metal.casing} castShadow receiveShadow />
      <Box position={[0, 13.65, 0]} size={[5.65, 0.2, 5.5]} material={metal.frameDark} castShadow receiveShadow />

      <InstancedCylinders placements={waterwallTubes} material={metal.casingLight} radialSegments={8} />
      <InstancedCylinders placements={boilerDowncomers} material={metal.casingDark} radialSegments={12} />
      <InstancedCylinders placements={waterwallHeaders} material={metal.casing} radialSegments={12} />

      <Box position={[0, 3.45, 2.7]} size={[3.3, 2.2, 0.18]} material={metal.casingDark} />
      <Box position={[0, 6.35, 2.7]} size={[3.3, 1.6, 0.18]} material={metal.casingDark} />
      {[-1.535, -0.56, 0.56, 1.535].map((x, index) => (
        <Box
          key={`firebox-divider-${index}`}
          position={[x, 5.1, 2.7]}
          size={[index === 0 || index === 3 ? 0.23 : 0.52, 1, 0.18]}
          material={metal.casingDark}
        />
      ))}
      {[-1.12, 0, 1.12].map((x) => (
        <Box key={`firebox-recess-${x}`} position={[x, 5.1, 2.67]} size={[0.6, 1, 0.04]} material={metal.furnace} />
      ))}
      <FireboxGlow drive={VISUAL_ONLY_DRIVE.boilerLoad} />
      <InstancedBoxes placements={burnerDoorFrames} material={metal.furnaceDoor} />
      <InstancedBoxes placements={burnerGrating} material={metal.frameDark} />
      <InstancedCylinders placements={burnerNozzleCollars} material={metal.furnaceDoor} radialSegments={16} />
      <InstancedCylinders placements={burnerNozzles} material={metal.casingDark} radialSegments={16} />
      {[-1.12, 0, 1.12].map((x) => (
        <Box key={`boiler-upper-burner-rail-${x}`} position={[x, 7.4, 2.83]} size={[0.72, 0.2, 0.1]} material={metal.frame} />
      ))}
      <InstancedCylinders placements={tubeBankBright} material={metal.casingLight} radialSegments={10} />
      <InstancedCylinders placements={tubeBankMain} material={metal.casing} radialSegments={10} />
      <InstancedCylinders placements={superheaterHeaders} material={metal.casingLight} radialSegments={10} />
      <Cylinder position={[0, 14.45, 0]} radius={0.82} length={7.1} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[0, 14.45, 0]} radius={0.7} length={7.45} material={metal.casing} rotation={[0, 0, Math.PI / 2]} />
      {[-3.68, 3.68].map((x) => (
        <BoltedFlange key={`steam-drum-flange-${x}`} position={[x, 14.45, 0]} radius={0.84} axis="x" thickness={0.14} />
      ))}
      {[-2.2, 2.2].map((x) => (
        <Box key={`steam-drum-saddle-${x}`} position={[x, 13.92, 0]} size={[0.42, 0.42, 1.55]} material={metal.frameDark} />
      ))}
      {[-2.45, 2.45].map((x) => (
        <group key={`roof-vessel-${x}`}>
          <Cylinder position={[x, 15.12, -1.75]} radius={0.28} length={2.2} material={metal.casingDark} rotation={[0, 0, Math.PI / 2]} radialSegments={24} />
          {[-1.12, 1.12].map((end) => (
            <BoltedFlange key={end} position={[x + end, 15.12, -1.75]} radius={0.34} axis="x" thickness={0.1} />
          ))}
          <Member from={[x, 14.7, -2.3]} to={[x, 15.65, -2.3]} width={0.08} material={metal.frameDark} />
        </group>
      ))}
      {[-2.8, 2.8].map((x) => (
        <Box key={x} position={[x, 15.1, 0]} size={[0.22, 0.55, 4.6]} material={metal.frameDark} />
      ))}
    </group>
  )
}

export default function BoilerSystem({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="BoilerSystem"
      position={[-3.5, 0, 0]}
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[11, 17.2, 8.5]} position={[0, 8.5, 0]} visible={selected} />
      <BoilerFrame />
      <BoilerFurnace />
      <group name="BoilerAccessStairs" position={[4.1, 0, 3.8]}>
        {Array.from({ length: 22 }, (_, i) => (
          <Box
            key={i}
            position={[i * 0.16 - 1.7, 0.5 + i * 0.29, 0]}
            size={[0.64, 0.1, 1.6]}
            material={metal.platform}
          />
        ))}
        <Member from={[-1.8, 0.4, 0.55]} to={[1.8, 6.9, 0.55]} width={0.08} material={metal.rail} />
        <Member from={[-1.8, 0.4, -0.55]} to={[1.8, 6.9, -0.55]} width={0.08} material={metal.rail} />
      </group>
    </group>
  )
}
