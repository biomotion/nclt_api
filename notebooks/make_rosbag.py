import sys
sys.path.append("..")

import velodyne_utils
import rosmsg_utils

import numpy as np
import scipy.interpolate
import os
from tqdm import tqdm
import rosbag
import rospy
import struct

def generate_sensors_bag(save_path, data_path):
    with rosbag.Bag(save_path, "w") as bag:
    
        # read velodyne hits
        pbar = tqdm(total=os.path.getsize(os.path.join(data_path, "velodyne_hits.bin")))
        with open(os.path.join(data_path, "velodyne_hits.bin"), "rb") as fd:
            count = 0
            while True:
                buf = fd.read(8)
                if buf == '':
                    print("eof")
                    break
                if not velodyne_utils.verify_magic(buf):
                    raise RuntimeWarning("magic not verified")
            
                num_hits, utime, pad = struct.unpack("<IQ4s", fd.read(16))

                pbar.set_postfix(order_dict = {"n_hist": num_hits, "utime": utime})
                buf = fd.read(velodyne_utils.dt_raw.itemsize*num_hits)
                points = velodyne_utils.read_hit_buf(buf)
                
                msg = rosmsg_utils.make_PointCloud2(
                    count, 
                    utime = float(utime)/1e6,
                    data = points
                )
                bag.write("/velodyne_hits", msg, t=rospy.Time(float(utime)/1e6))
                count += 1
                pbar.update(24 + velodyne_utils.dt_raw.itemsize*num_hits)
        pbar.close()
        
        gps_data = np.loadtxt(os.path.join(data_path, "gps.csv"), dtype=rosmsg_utils.dt_gps, delimiter=",")
        for i, g_data in enumerate(tqdm(gps_data)):
            msg = rosmsg_utils.make_GPS(g_data, i)
            bag.write("/garmin_gps", msg, t=rospy.Time(float(g_data['utime'])/1e6))
          
        rtk_data = np.loadtxt(os.path.join(data_path, "gps_rtk.csv"), dtype=rosmsg_utils.dt_gps, delimiter=",")
        for i, g_data in enumerate(tqdm(rtk_data)):
            msg = rosmsg_utils.make_GPS(g_data, i)
            bag.write("/rtk_gps", msg, t=rospy.Time(float(g_data['utime'])/1e6))
        
        imu_data = np.loadtxt(os.path.join(data_path, "ms25.csv"), dtype=rosmsg_utils.dt_imu, delimiter=",")
        imu_euler_data = np.loadtxt(os.path.join(data_path, "ms25_euler.csv"), dtype=rosmsg_utils.dt_imu_euler, delimiter=",")
        synced_euler_data = rosmsg_utils.interpolate_imu_euler(imu_euler_data, imu_data['utime'])
        
        for i, (i_data, euler_data) in enumerate(tqdm(zip(imu_data, synced_euler_data))):
            msg = rosmsg_utils.make_IMU(i, i_data['utime'], i_data['lin_acc'], i_data['ang_vel'], euler_data)
            bag.write("/microstrain_imu", msg, t=rospy.Time(float(i_data['utime'])/1e6))


if __name__ == "__main__":
    all_drives = [
        "2012-01-08",
        "2012-01-15",
        "2012-01-22",
        "2012-02-02",
        "2012-02-04",
        "2012-02-05",
        "2012-02-12",
        "2012-02-18",
        "2012-02-19",
        "2012-03-17",
        "2012-03-25",
        "2012-03-31",
        "2012-04-29",
        "2012-05-11",
        "2012-05-26",
        "2012-06-15",
        "2012-08-04",
        "2012-08-20",
        "2012-09-28",
        "2012-10-28",
        "2012-11-04",
        "2012-11-16",
        "2012-11-17",
        "2012-12-01",
        "2013-01-10",
        "2013-02-23",
        "2013-04-05",
    ]

    for drive in all_drives:
        print("converting {}...".format(drive))

        data_path = "/mnt/sshfs/ee904-3/Datasets_3rdParties/nclt/extracted/{}".format(drive)

        save_path = "/data/nclt_bags/{}_sensors.bag".format(drive)
        generate_sensors_bag(save_path, data_path)