export type EquipmentId =
  | 'CoalHandlingSystem'
  | 'CoalMill'
  | 'BoilerSystem'
  | 'Chimney'
  | 'TurbineSystem'
  | 'Generator'
  | 'CondenserSystem'
  | 'FeedwaterSystem'
  | 'FeedwaterPump'
  | 'CoolingSystem'
  | 'ProcessPipes'

export type EquipmentSelection = {
  id: EquipmentId
  instance?: number
}
