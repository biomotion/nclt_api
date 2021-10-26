import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointField, PointCloud2, Imu, NavSatFix, NavSatStatus
from geometry_msgs.msg import Quaternion, Vector3, PoseWithCovariance, Pose, Point
from nav_msgs.msg import Odometry
from tf_conversions import transformations
import scipy.interpolate


point_type_map = {
        np.int8 : 1,
        np.uint8 : 2,
        np.int16 : 3,
        np.uint16 : 4,
        np.int32 : 5,
        np.uint32 : 6,
        np.float32 : 7,
        np.float64 : 8,
}

dt_gps = np.dtype([
    ("utime", np.uint64),
    ("fix_mode", np.uint32),
    ("num_sat", np.uint32),
    ("lat", np.float64),
    ("lon", np.float64),
    ("alt", np.float64),
    ("track", np.float64),
    ("speed", np.float64)
])

gps_mode_map = {
    0: NavSatStatus.STATUS_NO_FIX,
    1: NavSatStatus.STATUS_NO_FIX,
    2: NavSatStatus.STATUS_SBAS_FIX,
    3: NavSatStatus.STATUS_GBAS_FIX,
}

dt_imu = np.dtype([
    ("utime", np.uint64),
    ("mag", np.float64, 3),
    ("lin_acc", np.float64, 3), # xyz
    ("ang_vel", np.float64, 3)  # rpy
])

dt_imu_euler = np.dtype([
    ("utime", np.uint64),
    ("roll", np.float64),
    ("pitch", np.float64),
    ("yaw", np.float64),
])

dt_pose = np.dtype([
    ("utime", np.uint64),
    ("xyz", np.float64, 3),
    ("rpy", np.float64, 3),
])

dt_cov = np.dtype([
    ("utime", np.uint64),
    ("cov", np.float64, (6, 6))
])


def make_PointFields(dt):
    # make list of PointField's from np.dtype object
    # assumes fields of dt all have size exactly 1

    point_fields = []
    for name in dt.names:
        t, offset = dt.fields[name]
        point_fields.append(
            PointField(name=name, offset=offset, datatype=point_type_map[t.type], count=1))
    return point_fields


def make_PointCloud2(seq, utime, data, frame_id="velodyne"):
    '''
    Input
    -------
    seq: index of message
    utime: message time in microseconds
    data: 1d array of point dtype containing [x, y, z, (intentisy, ring, ...)] data
    '''
    point_fields = make_PointFields(data.dtype)
    
    output = PointCloud2(
        header = Header(
            seq = seq,
            stamp = rospy.Time(float(utime)/1e6),
            frame_id=frame_id),
        height = 1,
        width = len(data),
        fields = point_fields,
        is_bigendian = False,
        point_step = data.dtype.itemsize,
        row_step = len(data) * data.dtype.itemsize,
        is_dense = False,
        data = data.tobytes()
    )
    return output


def make_GPS(data, seq, frame_id="world"):
    return NavSatFix(
        header = Header(
            seq = seq,
            stamp = rospy.Time(float(data['utime'])/1e6),
            frame_id=frame_id),
        status = NavSatStatus(
            status = gps_mode_map[data['fix_mode']],
            service = NavSatStatus.SERVICE_GPS
        ),
        latitude = data['lat'],
        longitude = data['lon'],
        altitude = data['alt'],
    )


def interpolate_imu_euler(euler_data, utime):
    rpy = np.vstack([
        euler_data['roll'],
        euler_data['pitch'],
        euler_data['yaw'],
    ]).T
    interp = scipy.interpolate.interp1d(
        euler_data['utime'], rpy, axis=0, 
        bounds_error=False, fill_value="extrapolate", assume_sorted=True)
    return interp(utime)


def make_IMU(seq, utime, lin_acc, ang_vel, orientation, frame_id="ms25"):
    cov = np.identity(3).reshape(-1) * 0.1
    q = transformations.quaternion_from_euler(*orientation)
    return Imu(
        header = Header(
            seq=seq,
            stamp=rospy.Time(float(utime)/1e6),
            frame_id=frame_id),
        orientation = Quaternion(
           x=q[0], y=q[1], z=q[2], w=q[3]
        ),
        orientation_covariance = cov,
        angular_velocity = Vector3(
            x=ang_vel[0], y=ang_vel[1], z=ang_vel[2]
        ),
        angular_velocity_covariance = cov,
        linear_acceleration = Vector3(
            x=lin_acc[0], y=lin_acc[1], z=lin_acc[2]
        ),
        linear_acceleration_covariance = cov,
    )


def read_Covariance6D(name):
    dt  = np.dtype([
        ("utime", np.uint64),
        ("cov", np.float64, (21,))
    ])
    I = np.triu_indices(6)
    cov_data = np.loadtxt(name, dtype=dt, delimiter=",")
    result = np.zeros(len(cov_data), dtype=dt_cov)
    result['utime'] = cov_data['utime']
    result['cov'][:,I[0], I[1]] = cov_data['cov']
    return result


def make_PoseWithCovariance(xyz, rpy, cov):
    q = transformations.quaternion_from_euler(*rpy)
    return PoseWithCovariance(
        pose = Pose(
            position = Point(
                x=xyz[0], y=xyz[1], z=xyz[2],
            ),
            orientation = Quaternion(
                x=q[0], y=q[1], z=q[2], w=q[3]
            )
        ),
        covariance=cov
    )


def make_Odometry(seq, utime, position, orientation, cov, frame_id="odom", child_frame_id="base_link"):
    return Odometry(
        header = Header(
            seq=seq,
            stamp=rospy.Time(float(utime)/1e6),
            frame_id=frame_id),
        child_frame_id = child_frame_id,
        pose = make_PoseWithCovariance(
            position, orientation, cov
        )
    )