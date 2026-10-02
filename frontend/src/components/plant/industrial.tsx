import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Edges } from '@react-three/drei'

export type Vec3 = [number, number, number]

const standard = (color: string, metalness = 0.45, roughness = 0.48) =>
  new THREE.MeshStandardMaterial({ color, metalness, roughness })

export const metal = {
  frame: standard('#899398', 0.68, 0.42),
  frameDark: standard('#59656a', 0.62, 0.5),
  platform: standard('#737e82', 0.42, 0.66),
  casing: standard('#a4adaf', 0.66, 0.38),
  casingLight: standard('#c2c8c9', 0.5, 0.42),
  casingDark: standard('#48565c', 0.64, 0.46),
  furnace: standard('#625d58', 0.32, 0.76),
  furnaceDoor: standard('#806b5d', 0.4, 0.58),
  coal: standard('#292d2e', 0.1, 0.9),
  belt: standard('#3d4446', 0.12, 0.86),
  generator: standard('#69777d', 0.55, 0.4),
  concrete: standard('#7b8588', 0.1, 0.86),
  water: standard('#66899a', 0.42, 0.38),
  waterReturn: standard('#8ba6af', 0.4, 0.4),
  steam: standard('#a96d58', 0.4, 0.42),
  feedwater: standard('#a99a78', 0.48, 0.44),
  gas: standard('#777f81', 0.52, 0.46),
  tower: standard('#bcc2c2', 0.18, 0.82),
  rail: standard('#d2d6d5', 0.52, 0.42),
}

export type BoxPlacement = { position: Vec3; size: Vec3; rotation?: Vec3 }
export type CylinderPlacement = { position: Vec3; radius: number; length: number; rotation?: Vec3 }

export function InstancedBoxes({
  placements,
  material = metal.frame,
}: {
  placements: BoxPlacement[]
  material?: THREE.Material
}) {
  const instances = useRef<THREE.InstancedMesh>(null)
  const transform = useMemo(() => new THREE.Object3D(), [])

  useLayoutEffect(() => {
    const mesh = instances.current
    if (!mesh) return
    placements.forEach(({ position, size, rotation = [0, 0, 0] }, index) => {
      transform.position.set(...position)
      transform.rotation.set(...rotation)
      transform.scale.set(...size)
      transform.updateMatrix()
      mesh.setMatrixAt(index, transform.matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [placements, transform])

  if (placements.length === 0) return null
  return (
    <instancedMesh ref={instances} args={[undefined, undefined, placements.length]} material={material} castShadow receiveShadow frustumCulled={false}>
      <boxGeometry args={[1, 1, 1]} />
    </instancedMesh>
  )
}

export function InstancedCylinders({
  placements,
  material = metal.casingLight,
  radialSegments = 16,
}: {
  placements: CylinderPlacement[]
  material?: THREE.Material
  radialSegments?: number
}) {
  const instances = useRef<THREE.InstancedMesh>(null)
  const transform = useMemo(() => new THREE.Object3D(), [])

  useLayoutEffect(() => {
    const mesh = instances.current
    if (!mesh) return
    placements.forEach(({ position, radius, length, rotation = [0, 0, 0] }, index) => {
      transform.position.set(...position)
      transform.rotation.set(...rotation)
      transform.scale.set(radius, length, radius)
      transform.updateMatrix()
      mesh.setMatrixAt(index, transform.matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
    mesh.computeBoundingSphere()
  }, [placements, transform])

  if (placements.length === 0) return null
  return (
    <instancedMesh ref={instances} args={[undefined, undefined, placements.length]} material={material} castShadow frustumCulled={false}>
      <cylinderGeometry args={[1, 1, 1, radialSegments]} />
    </instancedMesh>
  )
}

export function BoltedFlange({
  position,
  radius,
  axis = 'x',
  thickness = 0.13,
}: {
  position: Vec3
  radius: number
  axis?: 'x' | 'y' | 'z'
  thickness?: number
}) {
  const rotation = useMemo<Vec3>(
    () => axis === 'x' ? [0, 0, Math.PI / 2] : axis === 'z' ? [Math.PI / 2, 0, 0] : [0, 0, 0],
    [axis],
  )
  const bolts = useMemo(() => {
    const boltRadius = radius * 0.76
    return Array.from({ length: 8 }, (_, index) => {
      const angle = (index / 8) * Math.PI * 2
      const a = Math.cos(angle) * boltRadius
      const b = Math.sin(angle) * boltRadius
      const boltPosition: Vec3 = axis === 'x' ? [0, a, b] : axis === 'z' ? [a, b, 0] : [a, 0, b]
      return { position: boltPosition, radius: Math.max(0.025, radius * 0.055), length: thickness * 1.34, rotation }
    })
  }, [axis, radius, rotation, thickness])

  return (
    <group position={position}>
      <Cylinder position={[0, 0, 0]} radius={radius} length={thickness} material={metal.frameDark} rotation={rotation} radialSegments={32} />
      <InstancedCylinders placements={bolts} material={metal.casingLight} radialSegments={8} />
    </group>
  )
}

export function Box({
  position,
  size,
  material = metal.frame,
  rotation,
  castShadow = false,
  receiveShadow = false,
}: {
  position: Vec3
  size: Vec3
  material?: THREE.Material
  rotation?: Vec3
  castShadow?: boolean
  receiveShadow?: boolean
}) {
  return (
    <mesh
      position={position}
      rotation={rotation}
      material={material}
      castShadow={castShadow}
      receiveShadow={receiveShadow}
    >
      <boxGeometry args={size} />
    </mesh>
  )
}

export function Cylinder({
  position,
  radius,
  length,
  material = metal.casing,
  rotation = [0, 0, 0],
  radialSegments = 20,
}: {
  position: Vec3
  radius: number
  length: number
  material?: THREE.Material
  rotation?: Vec3
  radialSegments?: number
}) {
  return (
    <mesh position={position} rotation={rotation} material={material} castShadow>
      <cylinderGeometry args={[radius, radius, length, radialSegments]} />
    </mesh>
  )
}

export function Member({
  from,
  to,
  width = 0.14,
  depth = width,
  material = metal.frame,
}: {
  from: Vec3
  to: Vec3
  width?: number
  depth?: number
  material?: THREE.Material
}) {
  const { position, quaternion, length } = useMemo(() => {
    const start = new THREE.Vector3(...from)
    const end = new THREE.Vector3(...to)
    const direction = end.clone().sub(start)
    return {
      position: start.add(end).multiplyScalar(0.5),
      quaternion: new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.clone().normalize(),
      ),
      length: direction.length(),
    }
  }, [from, to])

  return (
    <mesh position={position} quaternion={quaternion} material={material}>
      <boxGeometry args={[width, length, depth]} />
    </mesh>
  )
}

export function Pipe({
  points,
  radius = 0.16,
  material = metal.steam,
  segments = 28,
}: {
  points: Vec3[]
  radius?: number
  material?: THREE.Material
  segments?: number
}) {
  const curve = useMemo(
    () =>
      new THREE.CatmullRomCurve3(
        points.map((point) => new THREE.Vector3(...point)),
        false,
        'centripetal',
      ),
    [points],
  )

  return (
    <mesh material={material} castShadow>
      <tubeGeometry args={[curve, segments, radius, 10, false]} />
    </mesh>
  )
}

export function SelectionBounds({
  size,
  position = [0, 0, 0],
  visible,
}: {
  size: Vec3
  position?: Vec3
  visible: boolean
}) {
  if (!visible) return null
  return (
    <mesh position={position} raycast={() => {}}>
      <boxGeometry args={size} />
      <meshBasicMaterial color="#000000" transparent opacity={0} depthWrite={false} />
      <Edges color="#61d6f3" linewidth={1.5} />
    </mesh>
  )
}

export function Foundation({
  position,
  size,
}: {
  position: Vec3
  size: Vec3
}) {
  return (
    <group position={position}>
      <Box position={[0, 0.16, 0]} size={size} material={metal.concrete} receiveShadow />
      <Box
        position={[0, 0.34, 0]}
        size={[size[0] * 0.96, 0.08, size[2] * 0.96]}
        material={metal.casingDark}
      />
    </group>
  )
}
