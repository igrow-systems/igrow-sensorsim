# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: observation.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import location_pb2 as location__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='observation.proto',
  package='com.igrow.observation',
  syntax='proto3',
  serialized_pb=_b('\n\x11observation.proto\x12\x15\x63om.igrow.observation\x1a\x0elocation.proto\"\xa6\x05\n\x0bObservation\x12@\n\x04type\x18\x01 \x01(\x0e\x32\x32.com.igrow.observation.Observation.ObservationType\x12\x10\n\x08sensorId\x18\x02 \x01(\t\x12\x11\n\ttimestamp\x18\x03 \x01(\x04\x12\x31\n\x08location\x18\x04 \x01(\x0b\x32\x1f.com.igrow.observation.Location\x12\x39\n\x04mode\x18\x05 \x01(\x0e\x32+.com.igrow.observation.Observation.ModeType\x12_\n\x14\x65nvSensorObservation\x18\x06 \x01(\x0b\x32\x41.com.igrow.observation.Observation.EnvironmentalSensorObservation\x12[\n\x17\x62\x61tteryLevelObservation\x18\x07 \x01(\x0b\x32:.com.igrow.observation.Observation.BatteryLevelObservation\x1a[\n\x1e\x45nvironmentalSensorObservation\x12\x13\n\x0btemperature\x18\x01 \x01(\x02\x12\x10\n\x08humidity\x18\x02 \x01(\x02\x12\x12\n\nirradiance\x18\x03 \x01(\x02\x1a/\n\x17\x42\x61tteryLevelObservation\x12\x14\n\x0c\x62\x61tteryLevel\x18\x01 \x01(\x02\"Q\n\x0fObservationType\x12\x11\n\rLOCATION_ONLY\x10\x00\x12\x18\n\x14\x45NVIRONMENTAL_SENSOR\x10\x01\x12\x11\n\rBATTERY_LEVEL\x10\x02\"#\n\x08ModeType\x12\x0b\n\x07PASSIVE\x10\x00\x12\n\n\x06\x41\x43TIVE\x10\x01\"H\n\x0cObservations\x12\x38\n\x0cobservations\x18\x01 \x03(\x0b\x32\".com.igrow.observation.ObservationB5\n\x1e\x63om.igrow.protobuf.observationB\x13ObservationProtoBufb\x06proto3')
  ,
  dependencies=[location__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_OBSERVATION_OBSERVATIONTYPE = _descriptor.EnumDescriptor(
  name='ObservationType',
  full_name='com.igrow.observation.Observation.ObservationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='LOCATION_ONLY', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ENVIRONMENTAL_SENSOR', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BATTERY_LEVEL', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=621,
  serialized_end=702,
)
_sym_db.RegisterEnumDescriptor(_OBSERVATION_OBSERVATIONTYPE)

_OBSERVATION_MODETYPE = _descriptor.EnumDescriptor(
  name='ModeType',
  full_name='com.igrow.observation.Observation.ModeType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PASSIVE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACTIVE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=704,
  serialized_end=739,
)
_sym_db.RegisterEnumDescriptor(_OBSERVATION_MODETYPE)


_OBSERVATION_ENVIRONMENTALSENSOROBSERVATION = _descriptor.Descriptor(
  name='EnvironmentalSensorObservation',
  full_name='com.igrow.observation.Observation.EnvironmentalSensorObservation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='temperature', full_name='com.igrow.observation.Observation.EnvironmentalSensorObservation.temperature', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='humidity', full_name='com.igrow.observation.Observation.EnvironmentalSensorObservation.humidity', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='irradiance', full_name='com.igrow.observation.Observation.EnvironmentalSensorObservation.irradiance', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=479,
  serialized_end=570,
)

_OBSERVATION_BATTERYLEVELOBSERVATION = _descriptor.Descriptor(
  name='BatteryLevelObservation',
  full_name='com.igrow.observation.Observation.BatteryLevelObservation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batteryLevel', full_name='com.igrow.observation.Observation.BatteryLevelObservation.batteryLevel', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=572,
  serialized_end=619,
)

_OBSERVATION = _descriptor.Descriptor(
  name='Observation',
  full_name='com.igrow.observation.Observation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='com.igrow.observation.Observation.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sensorId', full_name='com.igrow.observation.Observation.sensorId', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='com.igrow.observation.Observation.timestamp', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='location', full_name='com.igrow.observation.Observation.location', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mode', full_name='com.igrow.observation.Observation.mode', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='envSensorObservation', full_name='com.igrow.observation.Observation.envSensorObservation', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batteryLevelObservation', full_name='com.igrow.observation.Observation.batteryLevelObservation', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_OBSERVATION_ENVIRONMENTALSENSOROBSERVATION, _OBSERVATION_BATTERYLEVELOBSERVATION, ],
  enum_types=[
    _OBSERVATION_OBSERVATIONTYPE,
    _OBSERVATION_MODETYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=61,
  serialized_end=739,
)


_OBSERVATIONS = _descriptor.Descriptor(
  name='Observations',
  full_name='com.igrow.observation.Observations',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='observations', full_name='com.igrow.observation.Observations.observations', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=741,
  serialized_end=813,
)

_OBSERVATION_ENVIRONMENTALSENSOROBSERVATION.containing_type = _OBSERVATION
_OBSERVATION_BATTERYLEVELOBSERVATION.containing_type = _OBSERVATION
_OBSERVATION.fields_by_name['type'].enum_type = _OBSERVATION_OBSERVATIONTYPE
_OBSERVATION.fields_by_name['location'].message_type = location__pb2._LOCATION
_OBSERVATION.fields_by_name['mode'].enum_type = _OBSERVATION_MODETYPE
_OBSERVATION.fields_by_name['envSensorObservation'].message_type = _OBSERVATION_ENVIRONMENTALSENSOROBSERVATION
_OBSERVATION.fields_by_name['batteryLevelObservation'].message_type = _OBSERVATION_BATTERYLEVELOBSERVATION
_OBSERVATION_OBSERVATIONTYPE.containing_type = _OBSERVATION
_OBSERVATION_MODETYPE.containing_type = _OBSERVATION
_OBSERVATIONS.fields_by_name['observations'].message_type = _OBSERVATION
DESCRIPTOR.message_types_by_name['Observation'] = _OBSERVATION
DESCRIPTOR.message_types_by_name['Observations'] = _OBSERVATIONS

Observation = _reflection.GeneratedProtocolMessageType('Observation', (_message.Message,), dict(

  EnvironmentalSensorObservation = _reflection.GeneratedProtocolMessageType('EnvironmentalSensorObservation', (_message.Message,), dict(
    DESCRIPTOR = _OBSERVATION_ENVIRONMENTALSENSOROBSERVATION,
    __module__ = 'observation_pb2'
    # @@protoc_insertion_point(class_scope:com.igrow.observation.Observation.EnvironmentalSensorObservation)
    ))
  ,

  BatteryLevelObservation = _reflection.GeneratedProtocolMessageType('BatteryLevelObservation', (_message.Message,), dict(
    DESCRIPTOR = _OBSERVATION_BATTERYLEVELOBSERVATION,
    __module__ = 'observation_pb2'
    # @@protoc_insertion_point(class_scope:com.igrow.observation.Observation.BatteryLevelObservation)
    ))
  ,
  DESCRIPTOR = _OBSERVATION,
  __module__ = 'observation_pb2'
  # @@protoc_insertion_point(class_scope:com.igrow.observation.Observation)
  ))
_sym_db.RegisterMessage(Observation)
_sym_db.RegisterMessage(Observation.EnvironmentalSensorObservation)
_sym_db.RegisterMessage(Observation.BatteryLevelObservation)

Observations = _reflection.GeneratedProtocolMessageType('Observations', (_message.Message,), dict(
  DESCRIPTOR = _OBSERVATIONS,
  __module__ = 'observation_pb2'
  # @@protoc_insertion_point(class_scope:com.igrow.observation.Observations)
  ))
_sym_db.RegisterMessage(Observations)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\n\036com.igrow.protobuf.observationB\023ObservationProtoBuf'))
# @@protoc_insertion_point(module_scope)