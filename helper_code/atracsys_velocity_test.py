# from testbed import Testbed
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import numpy as np
from datetime import datetime
import pickle as pkl
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

kf = KalmanFilter(dim_x=6, dim_z=3)
dt = 1 / 100
kf.F = np.array([[1, 0, 0, dt, 0, 0],
                 [0, 1, 0, 0, dt, 0],
                 [0, 0, 1, 0, 0, dt],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0]])
kf.R = np.eye(3) * 8e-5
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-3, block_size=3, order_by_dim=False)
kf.x = np.array([0, 0, 0, 0, 0, 0])
kf_init = False


def callback(data):
    global v, pos, t, new_msg, kf_init, kf
    if not kf_init:
        kf.x = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z, 0, 0, 0])
        kf_init = True
    pos = data.pose.position
    pos = np.array([pos.x, pos.y, pos.z])
    t = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9
    t = datetime.fromtimestamp(t)
    kf.predict()
    kf.update(pos)
    v = np.linalg.norm([kf.x[3], kf.x[4], kf.x[5]])
    new_msg = True


if __name__ == "__main__":
    pos = None
    t = None
    new_msg = False
    v = None
    rospy.init_node("atracsys_velocity_test")
    sub = rospy.Subscriber("/atracsys/Carriage/measured_cp", PoseStamped, callback=callback)
    pub = rospy.Publisher("/atracsys/Carriage/measured_vel", Float32, queue_size=1)
    v_hist = []
    t_hist = []
    p_hist = []
    while not rospy.is_shutdown():
        if new_msg:
            v_hist.append(v)
            t_hist.append(t)
            p_hist.append(pos)
            msg = Float32()
            msg.data = v
            pub.publish(msg)
            new_msg = False

    v_hist = np.array(v_hist)
    t_hist = np.array(t_hist)
    p_hist = np.array(p_hist)
    with open("atracsys_velocity.pkl", "wb") as f:
        pkl.dump((p_hist, v_hist, t_hist), f)
