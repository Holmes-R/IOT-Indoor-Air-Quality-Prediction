import numpy as np
from filterpy.kalman import KalmanFilter

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
