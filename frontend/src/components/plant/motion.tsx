import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import type { Vec3 } from './industrial'

// Unitless visual-only drives. They are animation multipliers, not plant readings.
// Replace these defaults with normalized live values when an approved data mapping exists.
export const VISUAL_ONLY_DRIVE = {
  boilerLoad: 0.55,
  steamFlow: 0.52,
  turbineSpeed: 0.7,
  feedwaterFlow: 0.48,
  coalFlow: 0.42,
  coolingFlow: 0.4,
}

type FlowingParticlesProps = {
  points: Vec3[]
  color: string
  count?: number
  radius?: number
  speed?: number
  drive?: number
  offset?: Vec3
  faceted?: boolean
}

export function FlowingParticles({
  points,
  color,
  count = 9,
  radius = 0.055,
  speed = 0.07,
  drive = 0.5,
  offset = [0, 0, 0],
  faceted = false,
}: FlowingParticlesProps) {
  const instances = useRef<THREE.InstancedMesh>(null)
  const curve = useMemo(
    () => new THREE.CatmullRomCurve3(points.map((point) => new THREE.Vector3(...point)), false, 'centripetal'),
    [points],
  )
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const sample = useMemo(() => new THREE.Vector3(), [])
  const tangent = useMemo(() => new THREE.Vector3(), [])
  const offsetVector = useMemo(() => new THREE.Vector3(...offset), [offset])
  const travelAxis = useMemo(() => new THREE.Vector3(0, 1, 0), [])

  useFrame(({ clock }) => {
    const mesh = instances.current
    if (!mesh) return
    const time = clock.elapsedTime * speed * Math.max(0.15, drive)
    for (let i = 0; i < count; i += 1) {
      const progress = (i / count + time) % 1
      const point = curve.getPointAt(progress, sample).add(offsetVector)
      dummy.position.copy(point)
      dummy.scale.setScalar(radius * (0.84 + 0.16 * Math.sin(progress * Math.PI * 2)))
      if (faceted) {
        dummy.rotation.set(time * 0.8 + i, time * 0.55 + i * 0.7, time * 0.65 + i * 0.4)
      } else {
        dummy.quaternion.setFromUnitVectors(travelAxis, curve.getTangentAt(progress, tangent))
      }
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={instances} args={[undefined, undefined, count]} frustumCulled={false}>
      {faceted ? <dodecahedronGeometry args={[1, 0]} /> : <capsuleGeometry args={[0.44, 1.8, 3, 7]} />}
      <meshBasicMaterial color={color} toneMapped={false} />
    </instancedMesh>
  )
}

export function RisingVapor({
  count = 7,
  height = 2,
  radius = 0.26,
  speed = 0.55,
  drive = 0.5,
  drift = [0.3, 0, 0.25],
}: {
  count?: number
  height?: number
  radius?: number
  speed?: number
  drive?: number
  drift?: Vec3
}) {
  const instances = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const puffColor = useMemo(() => new THREE.Color(), [])
  const texture = useMemo(() => createVaporTexture(), [])

  useFrame(({ clock, camera }) => {
    const mesh = instances.current
    if (!mesh) return
    const elapsed = clock.elapsedTime * speed * Math.max(0.4, drive)
    for (let i = 0; i < count; i += 1) {
      const phase = (i / count + elapsed * 0.36) % 1
      const turbulence = Math.sin(elapsed * 2.2 + i * 1.73)
      const sway = Math.sin(elapsed * 0.95 + i * 1.41)
      const spread = radius * (0.12 + phase * 0.8)
      dummy.position.set(
        drift[0] * phase + sway * spread + turbulence * radius * 0.2,
        phase * height + Math.sin(elapsed * 1.4 + i * 2.3) * height * 0.012,
        drift[2] * phase + Math.cos(elapsed * 0.74 + i * 1.17) * spread * 0.78,
      )
      dummy.quaternion.copy(camera.quaternion)
      dummy.rotation.z += sway * 0.055
      const irregular = 1 + turbulence * 0.08
      dummy.scale.set(
        radius * (0.92 + phase * 1.9) * irregular,
        radius * (1.5 + phase * 2.25) * (1 - turbulence * 0.035),
        1,
      )
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      const fade = Math.max(0.12, 0.78 - phase * 0.55 + Math.sin(elapsed + i) * 0.035)
      puffColor.setRGB(fade, fade, fade)
      mesh.setColorAt(i, puffColor)
    }
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh ref={instances} args={[undefined, undefined, count]} frustumCulled={false}>
      <planeGeometry args={[1, 2]} />
      <meshBasicMaterial map={texture} color="#d5e0e3" transparent opacity={0.38} depthWrite={false} toneMapped={false} />
    </instancedMesh>
  )
}

function createVaporTexture() {
  const canvas = document.createElement('canvas')
  canvas.width = 128
  canvas.height = 192
  const context = canvas.getContext('2d')
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter
  if (!context) return texture

  let seed = 417
  const random = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return seed / 0x7fffffff
  }
  context.fillStyle = '#fff'
  context.filter = 'blur(15px)'
  for (let i = 0; i < 18; i += 1) {
    const x = 30 + random() * 68
    const y = 24 + random() * 145
    const rx = 14 + random() * 27
    const ry = 14 + random() * 25
    context.beginPath()
    context.ellipse(x, y, rx, ry, (random() - 0.5) * 0.8, 0, Math.PI * 2)
    context.fill()
  }
  context.filter = 'blur(5px)'
  context.globalAlpha = 0.5
  for (let i = 0; i < 9; i += 1) {
    const x = 32 + random() * 64
    const y = 28 + random() * 136
    context.beginPath()
    context.ellipse(x, y, 8 + random() * 16, 11 + random() * 20, random(), 0, Math.PI * 2)
    context.fill()
  }
  context.globalAlpha = 1
  context.filter = 'none'
  context.globalCompositeOperation = 'destination-in'
  const fade = context.createLinearGradient(0, 0, 0, canvas.height)
  fade.addColorStop(0, 'rgba(255,255,255,0)')
  fade.addColorStop(0.16, 'rgba(255,255,255,0.72)')
  fade.addColorStop(0.78, 'rgba(255,255,255,0.92)')
  fade.addColorStop(1, 'rgba(255,255,255,0)')
  context.fillStyle = fade
  context.fillRect(0, 0, canvas.width, canvas.height)
  texture.needsUpdate = true
  return texture
}

function createFlameTexture(seed: number, core = false) {
  const canvas = document.createElement('canvas')
  canvas.width = 96
  canvas.height = 144
  const context = canvas.getContext('2d')
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter
  if (!context) return texture

  let state = seed >>> 0
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 4294967296
  }
  const tongues = core ? 2 : 4
  context.globalCompositeOperation = 'lighter'
  context.shadowBlur = core ? 8 : 13
  context.shadowColor = core ? '#fff1a1' : '#ff531b'

  for (let i = 0; i < tongues; i += 1) {
    const width = (core ? 13 : 22) + random() * (core ? 10 : 18)
    const height = 64 + random() * (core ? 42 : 68)
    const center = 16 + random() * 64
    const tip = center + (random() - 0.5) * width * 0.9
    const baseY = 142 + random() * 2
    const tipY = baseY - height
    const gradient = context.createLinearGradient(center, baseY, tip, tipY)
    if (core) {
      gradient.addColorStop(0, 'rgba(255,255,224,0.96)')
      gradient.addColorStop(0.28, 'rgba(255,239,157,0.95)')
      gradient.addColorStop(0.68, 'rgba(255,182,58,0.74)')
      gradient.addColorStop(1, 'rgba(255,171,35,0)')
    } else {
      gradient.addColorStop(0, 'rgba(255,222,122,0.78)')
      gradient.addColorStop(0.25, 'rgba(255,174,54,0.9)')
      gradient.addColorStop(0.65, 'rgba(255,86,24,0.76)')
      gradient.addColorStop(1, 'rgba(216,41,12,0)')
    }
    context.fillStyle = gradient
    context.beginPath()
    context.moveTo(center - width * 0.55, baseY)
    context.bezierCurveTo(center - width * 0.78, baseY - height * 0.28, tip - width * 0.56, tipY + height * 0.17, tip, tipY)
    context.bezierCurveTo(tip + width * 0.42, tipY + height * 0.2, center + width * 0.82, baseY - height * 0.31, center + width * 0.55, baseY)
    context.closePath()
    context.fill()
  }

  texture.needsUpdate = true
  return texture
}

export function BurnerFlames({ drive = 0.55 }: { drive?: number }) {
  const flames = useRef<Array<THREE.Sprite | null>>([])
  const licks = useRef<Array<THREE.Sprite | null>>([])
  const cores = useRef<Array<THREE.Sprite | null>>([])
  const halos = useRef<Array<THREE.MeshBasicMaterial | null>>([])
  const outerTextures = useMemo(() => [createFlameTexture(17), createFlameTexture(83), createFlameTexture(149)], [])
  const coreTextures = useMemo(() => [createFlameTexture(29, true), createFlameTexture(101, true), createFlameTexture(197, true)], [])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime * (0.85 + drive * 0.5)
    flames.current.forEach((flame, index) => {
      if (!flame) return
      const wave = Math.sin(t * (7.2 + index * 0.13) + index * 1.9)
      const flutter = Math.sin(t * (12.1 + index * 0.27) + index * 0.83)
      flame.scale.set(0.37 * (0.92 + wave * 0.07), 0.64 * (0.9 + flutter * 0.08 + drive * 0.08), 1)
      flame.position.x = Math.sin(t * 3.2 + index * 1.7) * 0.018
      flame.position.y = Math.sin(t * 6.1 + index * 0.9) * 0.014
      flame.material.opacity = 0.62 + drive * 0.09 + 0.07 * Math.sin(t * 5.6 + index * 1.2)
      flame.material.rotation = Math.sin(t * 3.8 + index * 1.7) * 0.035
    })
    licks.current.forEach((lick, index) => {
      if (!lick) return
      const wave = Math.sin(t * 8.4 + index * 1.31)
      const flutter = Math.sin(t * 13.7 + index * 0.77)
      lick.scale.set(0.2 * (0.92 + wave * 0.12), 0.42 * (0.88 + flutter * 0.1), 1)
      lick.position.x = (index % 2 === 0 ? -1 : 1) * (0.06 + wave * 0.018)
      lick.position.y = 0.035 + flutter * 0.024
      lick.material.opacity = 0.24 + drive * 0.05 + 0.035 * wave
      lick.material.rotation = Math.sin(t * 4.8 + index) * 0.05
    })
    cores.current.forEach((core, index) => {
      if (!core) return
      const pulse = Math.sin(t * 8.7 + index * 2.1)
      core.scale.set(0.18 * (0.9 + pulse * 0.08), 0.42 * (0.92 + pulse * 0.07), 1)
      core.material.opacity = 0.53 + drive * 0.1 + pulse * 0.08
    })
    halos.current.forEach((material, index) => {
      if (material) material.opacity = 0.16 + (0.035 + drive * 0.025) * (0.5 + 0.5 * Math.sin(t * 4.7 + index * 1.6))
    })
  })

  return (
    <group name="VisualOnlyFurnaceCombustion">
      {[-1.12, 0, 1.12].map((x, index) => (
        <group key={x} position={[x, 5.1, 2.965]}>
          <mesh position={[0, 0, 0.012]}>
          <planeGeometry args={[0.4, 0.66]} />
            <meshBasicMaterial
              ref={(material) => { halos.current[index] = material }}
              color="#ff4b10"
              transparent
              opacity={0.19}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <sprite
            ref={(sprite) => { flames.current[index] = sprite }}
            position={[(index - 1) * 0.025, 0, 0.035]}
            scale={[0.37, 0.64, 1]}
          >
            <spriteMaterial map={outerTextures[index]} transparent opacity={0.67} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
          </sprite>
          <sprite
            ref={(sprite) => { licks.current[index * 2] = sprite }}
            position={[-0.06, 0.035, 0.039]}
            scale={[0.2, 0.42, 1]}
          >
            <spriteMaterial map={outerTextures[(index + 1) % outerTextures.length]} transparent opacity={0.28} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
          </sprite>
          <sprite
            ref={(sprite) => { licks.current[index * 2 + 1] = sprite }}
            position={[0.06, 0.035, 0.04]}
            scale={[0.2, 0.42, 1]}
          >
            <spriteMaterial map={outerTextures[(index + 2) % outerTextures.length]} transparent opacity={0.28} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
          </sprite>
          <sprite
            ref={(sprite) => { cores.current[index] = sprite }}
            position={[(index - 1) * -0.018, -0.11, 0.045]}
            scale={[0.18, 0.42, 1]}
          >
            <spriteMaterial map={coreTextures[index]} transparent opacity={0.65} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
          </sprite>
        </group>
      ))}
    </group>
  )
}

export function RotatingPumpCoupling({ drive = 0.48 }: { drive?: number }) {
  const rotor = useRef<THREE.Group>(null)
  useFrame((_, delta) => {
    if (rotor.current) rotor.current.rotation.x += delta * (1.8 + drive * 1.5)
  })

  return (
    <group ref={rotor} position={[1.22, 0.72, 0]} name="VisualOnlyPumpCoupling">
      <mesh rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.2, 0.2, 0.08, 16]} />
        <meshStandardMaterial color="#74838a" metalness={0.7} roughness={0.4} />
      </mesh>
      {[0, 1, 2, 3].map((blade) => (
        <mesh key={blade} position={[0, Math.cos((blade * Math.PI) / 2) * 0.12, Math.sin((blade * Math.PI) / 2) * 0.12]} rotation={[(blade * Math.PI) / 2, 0, 0]}>
          <boxGeometry args={[0.045, 0.25, 0.045]} />
          <meshStandardMaterial color="#aab2b5" metalness={0.6} roughness={0.42} />
        </mesh>
      ))}
    </group>
  )
}
