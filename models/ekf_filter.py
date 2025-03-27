import numpy as np
from filterpy.kalman import KalmanFilter


'''This file applies an Extended Kalman Filter (EKF) to smooth sensor data.'''

"""
State and measurement dimensions.

Initial state from the first data point.

State transition matrix.

Measurement matrix.

High initial uncertainty.

Measurement noise covariance.

Process noise covariance.
"""


def apply_ekf(sensor_data):
    ekf = KalmanFilter(dim_x=1, dim_z=1)
    ekf.x = np.array([sensor_data[0]])  
    ekf.F = np.array([[1]])
    ekf.H = np.array([[1]])
    ekf.P *= 1000  
    ekf.R = 5  
    ekf.Q = 0.1  

    filtered_data = []
    for value in sensor_data:
        ekf.predict()
        ekf.update(value)
        filtered_data.append(ekf.x[0])

    return np.array(filtered_data)
