import { useMemo } from 'react'
import * as THREE from 'three'
import { Box, Cylinder, Foundation, InstancedBoxes, InstancedCylinders, InstancedMembers, SelectionBounds, metal } from './industrial'
import type { BoxPlacement, CylinderPlacement, MemberPlacement, Vec3 } from './industrial'
import { RisingVapor, RotatingPumpCoupling, VISUAL_ONLY_DRIVE } from './motion'

const towerProfile: [number, number][] = [
  [4.55, 0.25],
  [4.45, 1.15],
  [4.05, 3.05],
  [3.5, 5.2],
  [2.8, 7.3],
  [2.38, 9.15],
  [2.42, 10.6],
  [2.68, 12.15],
  [2.98, 13.6],
  [3.13, 14.35],
]

const towerRibSegments: CylinderPlacement[] = Array.from({ length: 28 }, (_, index) => {
  const angle = (index / 28) * Math.PI * 2
  return towerProfile.slice(1).map(([,], segment) => {
    const [startRadius, startY] = towerProfile[segment]
    const [endRadius, endY] = towerProfile[segment + 1]
    const start = new THREE.Vector3(Math.cos(angle) * (startRadius + 0.025), startY, Math.sin(angle) * (startRadius + 0.025))
    const end = new THREE.Vector3(Math.cos(angle) * (endRadius + 0.025), endY, Math.sin(angle) * (endRadius + 0.025))
    const direction = end.clone().sub(start)
    const rotation = new THREE.Euler().setFromQuaternion(
      new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize()),
    )
    return {
      position: [
        (start.x + end.x) / 2,
        (start.y + end.y) / 2,
        (start.z + end.z) / 2,
      ] as Vec3,
      radius: 0.035,
      length: direction.length(),
      rotation: [rotation.x, rotation.y, rotation.z] as Vec3,
    }
  })
}).flat()

const towerFrame: MemberPlacement[] = Array.from({ length: 20 }, (_, index) => {
  const angle = (index / 20) * Math.PI * 2
  return [
    {
      from: [Math.cos(angle) * 4.48, 0.6, Math.sin(angle) * 4.48] as Vec3,
      to: [Math.cos(angle) * 3.55, 4.8, Math.sin(angle) * 3.55] as Vec3,
      width: 0.1,
      depth: 0.1,
    },
    {
      from: [Math.cos(angle) * 4.45, 1.3, Math.sin(angle) * 4.45] as Vec3,
      to: [Math.cos(angle + Math.PI / 10) * 4.05, 3.1, Math.sin(angle + Math.PI / 10) * 4.05] as Vec3,
      width: 0.065,
      depth: 0.065,
    },
  ]
}).flat()

function towerRadiusAt(y: number) {
  const upper = towerProfile.findIndex((point) => point[1] >= y)
  if (upper <= 0) return towerProfile[0][0]
  const [ra, ya] = towerProfile[upper - 1]
  const [rb, yb] = towerProfile[upper]
  const t = (y - ya) / (yb - ya)
  return ra + (rb - ra) * t
}

function CoolingTower() {
  const geometry = useMemo(
    () =>
      new THREE.LatheGeometry(
        towerProfile.map(([radius, y]) => new THREE.Vector2(radius, y)),
        52,
      ),
    [],
  )
  const innerGeometry = useMemo(
    () => new THREE.LatheGeometry(towerProfile.map(([radius, y]) => new THREE.Vector2(radius - 0.18, y)), 52),
    [],
  )
  const radialRoofBeams = useMemo<BoxPlacement[]>(() => Array.from({ length: 16 }, (_, index) => {
    const angle = (index / 16) * Math.PI * 2
    const length = 2.72
    return {
      position: [Math.cos(angle) * length / 2, 13.78, Math.sin(angle) * length / 2],
      size: [length, 0.09, 0.085],
      rotation: [0, -angle, 0],
    }
  }), [])

  return (
    <group name="CoolingTower" position={[11.2, 0, -9.5]}>
      <Foundation position={[0, 0.1, 0]} size={[10.5, 0.35, 10.5]} />
      <mesh geometry={geometry} material={metal.tower} castShadow receiveShadow />
      <mesh geometry={innerGeometry} material={metal.casingLight} />
      <mesh position={[0, 14.35, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.casingLight}>
        <torusGeometry args={[3.04, 0.1, 8, 64]} />
      </mesh>
      <group name="VisualOnlyCoolingPlume" position={[0, 14.42, 0]}>
        <RisingVapor count={16} height={4.2} radius={0.62} speed={0.42} drive={VISUAL_ONLY_DRIVE.coolingFlow} />
      </group>
      <Cylinder position={[0, 0.55, 0]} radius={4.55} length={0.56} material={metal.frameDark} radialSegments={52} />
      {[1.25, 3.05, 5.2, 7.3, 9.15, 10.6, 12.15, 13.6].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.casingLight}>
          <torusGeometry args={[towerRadiusAt(y), y === 7.3 ? 0.075 : 0.035, 7, 52]} />
        </mesh>
      ))}
      <mesh position={[0, 13.78, 0]} rotation={[Math.PI / 2, 0, 0]} material={metal.frameDark}>
        <torusGeometry args={[2.78, 0.065, 7, 52]} />
      </mesh>
      <InstancedBoxes placements={radialRoofBeams} material={metal.frameDark} />
      <InstancedCylinders placements={towerRibSegments} material={metal.casingLight} radialSegments={6} />
      <InstancedMembers placements={towerFrame} material={metal.frameDark} />
      {[-1, 1].map((side) => (
        <group key={side} position={[side * 3.3, 2.2, 3.7]}>
          <Box position={[0, 0, 0]} size={[1.5, 3.2, 0.12]} material={metal.frameDark} />
          {[-0.6, 0.6].map((x) => (
            <Box key={x} position={[x, 0, 0.09]} size={[0.12, 3.4, 0.08]} material={metal.casingLight} />
          ))}
          {[-1, 0, 1].map((y) => (
            <Box key={y} position={[0, y, 0.11]} size={[1.5, 0.1, 0.08]} material={metal.frame} />
          ))}
        </group>
      ))}
    </group>
  )
}

function CoolingPump({ position, index }: { position: [number, number, number]; index: number }) {
  return (
    <group position={position} name={`CoolingWaterPump-${index}`}>
      <Foundation position={[0, 0.12, 0]} size={[2.4, 0.32, 2]} />
      <Box position={[-0.42, 0.72, 0]} size={[1.1, 0.88, 1.05]} material={metal.casingDark} />
      <Cylinder position={[0.45, 0.72, 0]} radius={0.43} length={1.05} material={metal.casingLight} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[1, 0.72, 0]} radius={0.3} length={0.22} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} />
      <Cylinder position={[-0.97, 0.72, 0]} radius={0.28} length={0.18} material={metal.frameDark} rotation={[0, 0, Math.PI / 2]} />
      <RotatingPumpCoupling drive={VISUAL_ONLY_DRIVE.coolingFlow} />
      <Box position={[-0.42, 1.22, 0]} size={[0.55, 0.14, 0.92]} material={metal.casing} />
    </group>
  )
}

export default function CoolingSystem({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="CoolingSystem"
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[12.2, 15.2, 13.4]} position={[11.2, 7.5, -8.2]} visible={selected} />
      <CoolingTower />
      <group name="CoolingWaterEquipment">
        <CoolingPump position={[7.2, 0, -3.3]} index={1} />
        <CoolingPump position={[10, 0, -3.3]} index={2} />
        <Box position={[8.6, 0.1, -3.3]} size={[6.4, 0.22, 2.5]} material={metal.concrete} />
      </group>
    </group>
  )
}
