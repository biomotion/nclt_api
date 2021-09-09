#!/usr/bin/env python

import struct
import numpy as np

ring_ids_hdl32e = np.array([0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8,
                     24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31])

dt_raw = np.dtype([
    ('x', np.uint16),
    ('y', np.uint16),
    ('z', np.uint16),
    ('i', np.uint8),
    ('ring', np.uint8),
])

dt_point = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('i', np.uint8),
    ('ring', np.uint8)
])

def verify_magic(s):
    magic = 44444
    m = struct.unpack('<HHHH', s)
    return len(m)>=4 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic

def raw_data_to_points(np_data):
    result = np.empty(len(np_data), dtype=dt_point)
    result['x'] = np_data['x'].astype(np.float32) * 0.005 - 100.
    result['y'] = np_data['y'].astype(np.float32) * 0.005 - 100.
    result['z'] = np_data['z'].astype(np.float32) * 0.005 - 100.
    result['i'] = np_data['i']
    result['ring'] = ring_ids_hdl32e[np_data['ring']]
    return result

def read_sync_bin_file(name):
    data = np.fromfile(name, dtype=dt_raw)
    points = raw_data_to_points(data)
    return points

def read_hit_buf(buffer):
    data = np.frombuffer(buffer, dtype=dt_raw)
    points = raw_data_to_points(data)
    return points
