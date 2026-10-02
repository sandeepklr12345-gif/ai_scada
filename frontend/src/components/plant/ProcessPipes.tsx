import { Box, Member, Pipe, SelectionBounds, metal } from './industrial'
import type { Vec3 } from './industrial'
import { FlowingParticles, VISUAL_ONLY_DRIVE } from './motion'

export const MAIN_STEAM_ROUTE: Vec3[] = [
  [0.05, 14.45, 0],
  [0.42, 14.45, 0.3],
  [0.42, 13.45, 2.4],
  [0.42, 5.1, 2.4],
  [0.78, 4.05, 1.85],
  [3.9, 3.65, 1.1],
]

export const CONDENSER_TO_PUMPS_ROUTE: Vec3[] = [
  [3.1, 0.98, 5.65],
  [3.05, 0.86, 6.35],
  [2.55, 0.84, 7.15],
  [2.35, 0.9, 8.05],
  [1.25, 0.96, 8.2],
]

export const FEEDWATER_RETURN_ROUTE: Vec3[] = [
  [1.8, 1.05, 8.45],
  [1.9, 1.55, 8.72],
  [0.65, 2.55, 8.55],
  [-1.15, 3.45, 7.55],
  [-2.35, 5.2, 5.35],
  [-3.2, 8.25, 3.0],
  [-3.2, 10.15, 2.7],
]

const TURBINE_EXHAUST_ROUTE: Vec3[] = [
  [8.92, 3.5, 0],
  [9.1, 3.1, 0.35],
  [9.1, 2.55, 1.3],
  [8.3, 2.28, 2.35],
  [7.2, 2.25, 2.85],
]

const COOLING_SUPPLY_ROUTE: Vec3[] = [
  [11, 0.72, -3.3],
  [10.8, 0.72, -4.55],
  [8.2, 0.72, -4.55],
  [8.2, 0.72, -3.3],
  [9.45, 0.72, -2.5],
  [9.45, 0.72, -0.5],
  [9.45, 0.72, 1.25],
  [8.8, 0.72, 2.05],
  [8.2, 0.72, 2.75],
]

const COOLING_RETURN_ROUTE: Vec3[] = [
  [7.45, 1.1, 5.9],
  [10.2, 1.05, 6.55],
  [15.5, 1.0, 6.55],
  [16.7, 0.95, 4.1],
  [16.7, 0.95, 0.8],
  [16.7, 0.9, -2.5],
  [15.9, 0.88, -4.15],
  [15.9, 0.88, -5.1],
  [12.1, 0.88, -5.1],
]

const feedwaterSupports: [number, number, number][] = [
  [-0.7, 3.0, 8],
  [-2.2, 5.3, 5.5],
  [-3.1, 8.1, 3.2],
]

export default function ProcessPipes({
  selected,
  onSelect,
}: {
  selected: boolean
  onSelect: () => void
}) {
  return (
    <group
      name="ProcessPipes"
      onClick={(event) => {
        event.stopPropagation()
        onSelect()
      }}
    >
      <SelectionBounds size={[34, 19, 26]} position={[0, 9, -1]} visible={selected} />
      <group name="MainSteamPiping">
        <Pipe
          points={MAIN_STEAM_ROUTE}
          radius={0.25}
          material={metal.steam}
          segments={54}
        />
        <FlowingParticles points={MAIN_STEAM_ROUTE} color="#ffd2a4" count={9} radius={0.052} speed={0.12} drive={VISUAL_ONLY_DRIVE.steamFlow} offset={[0, 0.13, 0.3]} />
        <Box position={[0.42, 13.45, 2.4]} size={[0.6, 0.66, 0.66]} material={metal.frameDark} />
        <Member from={[0.42, 0.2, 3.05]} to={[0.42, 13.3, 3.05]} width={0.12} material={metal.frameDark} />
        <Box position={[0.42, 0.32, 3.05]} size={[0.7, 0.16, 0.7]} material={metal.frame} />
        {[4, 7, 10, 13].map((y) => (
          <Member
            key={y}
            from={[0.42, y, 2.4]}
            to={[0.42, y, 3.05]}
            width={0.1}
            depth={0.1}
            material={metal.frame}
          />
        ))}
      </group>
      <group name="TurbineExhaustToCondenser">
        <Pipe
          points={TURBINE_EXHAUST_ROUTE}
          radius={0.31}
          material={metal.casingDark}
          segments={28}
        />
        <FlowingParticles points={TURBINE_EXHAUST_ROUTE} color="#d9dfe0" count={7} radius={0.055} speed={0.09} drive={VISUAL_ONLY_DRIVE.steamFlow} offset={[0, 0.22, 0.18]} />
        <Member from={[9.1, 0.45, 0.85]} to={[9.1, 2.54, 1.35]} width={0.11} material={metal.frameDark} />
      </group>
      <group name="CondenserToFeedwaterPumps">
        <Pipe
          points={CONDENSER_TO_PUMPS_ROUTE}
          radius={0.17}
          material={metal.feedwater}
        />
        <FlowingParticles points={CONDENSER_TO_PUMPS_ROUTE} color="#8ddcff" count={7} radius={0.05} speed={0.1} drive={VISUAL_ONLY_DRIVE.feedwaterFlow} offset={[0, 0.18, 0.1]} />
        {[-2.6, -0.45, 1.7].map((x) => (
          <Pipe
            key={x}
            points={[[x, 0.72, 8.2], [x, 0.96, 8.2]]}
            radius={0.11}
            material={metal.feedwater}
            segments={8}
          />
        ))}
      </group>
      <group name="FeedwaterPumpDischargeConnections">
        {[-0.82, 1.33, 3.48].map((x) => (
          <Pipe
            key={x}
            points={[[x, 0.72, 8.2], [x, 0.94, 8.25], [x, 1.05, 8.45]]}
            radius={0.11}
            material={metal.feedwater}
            segments={10}
          />
        ))}
      </group>
      <group name="FeedwaterReturnToBoiler">
        <Pipe
          points={FEEDWATER_RETURN_ROUTE}
          radius={0.19}
          material={metal.feedwater}
          segments={44}
        />
        <FlowingParticles points={FEEDWATER_RETURN_ROUTE} color="#c9e7ef" count={10} radius={0.052} speed={0.095} drive={VISUAL_ONLY_DRIVE.feedwaterFlow} offset={[0, 0.18, 0.12]} />
        {feedwaterSupports.map((position, i) => (
          <group key={i}>
            <Member from={[position[0], 0.35, position[2]]} to={[position[0], position[1], position[2]]} width={0.1} material={metal.frameDark} />
            <Box position={[position[0], 0.42, position[2]]} size={[0.7, 0.16, 0.7]} material={metal.frame} />
          </group>
        ))}
      </group>
      <group name="CoolingWaterCircuit">
        <Pipe
          points={[
            [10.5, 0.72, -5.05],
            [9.8, 0.72, -4.65],
            [9.03, 0.72, -3.3],
          ]}
          radius={0.24}
          material={metal.water}
          segments={20}
        />
        <Pipe
          points={[
            [9.03, 0.72, -3.3],
            [7.2, 0.72, -4.55],
            [6.23, 0.72, -3.3],
          ]}
          radius={0.21}
          material={metal.water}
          segments={22}
        />
        <Pipe
          points={COOLING_SUPPLY_ROUTE.slice(0, 4)}
          radius={0.21}
          material={metal.water}
          segments={22}
        />
        <Pipe
          points={COOLING_SUPPLY_ROUTE.slice(3)}
          radius={0.24}
          material={metal.water}
          segments={34}
        />
        <FlowingParticles points={COOLING_SUPPLY_ROUTE} color="#70bde0" count={10} radius={0.06} speed={0.075} drive={VISUAL_ONLY_DRIVE.coolingFlow} offset={[0, 0.16, 0.09]} />
        <Pipe
          points={COOLING_RETURN_ROUTE}
          radius={0.24}
          material={metal.waterReturn}
          segments={54}
        />
        <FlowingParticles points={COOLING_RETURN_ROUTE} color="#9bcfe2" count={12} radius={0.06} speed={0.065} drive={VISUAL_ONLY_DRIVE.coolingFlow} offset={[0, 0.2, 0.09]} />
        {[
          { x: 8.2, z: -4.55 },
          { x: 10.8, z: -4.55 },
          { x: 9.45, z: -1 },
          { x: 8.8, z: 2.05 },
          { x: 16.4, z: 0.8 },
          { x: 16.4, z: -4 },
          { x: 14.2, z: 6.55 },
        ].map(({ x, z }) => (
          <group key={`${x}-${z}`}>
            <Member from={[x, 0.35, z]} to={[x, 0.65, z]} width={0.11} material={metal.frameDark} />
            <Box position={[x, 0.42, z]} size={[0.68, 0.14, 0.68]} material={metal.frame} />
          </group>
        ))}
      </group>
      <group name="CombustionGasPath">
        <Pipe
          points={[
            [-4.7, 12.9, -2.65],
            [-5.0, 13.75, -2.5],
            [-5.7, 14.15, -2.4],
            [-6.4, 14.15, -2.4],
          ]}
          radius={0.38}
          material={metal.gas}
          segments={25}
        />
      </group>
    </group>
  )
}
