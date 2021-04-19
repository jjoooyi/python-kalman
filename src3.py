'''
1,Input parameters 매개변수
x : ndarray (dim_x, 1), default = [0,0,0…0]  Represents the state vector that the filter needs to estimate
P : ndarray (dim_x, dim_x), default eye(dim_x)  Represents the covariance matrix
Q : ndarray (dim_x, dim_x), default eye(dim_x)  Represents process noise (system noise)
R : ndarray (dim_z, dim_z), default eye(dim_x)  Indicates measurement noise
H : ndarray (dim_z, dim_x)  Represents the measurement equation
F : ndarray (dim_x, dim_x)  Represents the state transition equation
B : ndarray (dim_x, dim_u), default 0  Represents the control transfer matrix

2, Optional parameters
alpha : float
Assign a value > 1.0 to turn this into a fading memory filter.

3.See parameters
K :  ndarray Kalman gain
y :  ndarray residual, the difference between the actual measurement and the current estimated state mapped to the measurement space =(z - Hx)
S :  ndarray system uncertainty is mapped to measurement space =HPH’ + R 。
likelihood : float  The likelihood of the last measurement update.
log_likelihood : float  The log likelihood of the last measurement update.

...칼만 필터code...

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
#filter
kf = KalmanFilter(dim_x=2, dim_z=1)  #dim_x: hidden state size, dim_z: measurement size
#Define parameters	 
kf.x = x #Initial state [position, speed]
kf.F = F #State transition matrix
kf.H = np.array([[1.,0.]])  #Measurement matrix
kf.P = P #Initial state covariance
kf.R = R #Measurement noise
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-2) #Process (system) noise

z = get_sensor_reading()  #Input measurement
f.predict() #prediction
f.update(z) #Update
do_something_with_estimate (f.x)   #f.xUpdated status
'''
# 칼만필터 샘플 코드
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
np.random.seed(2)


def demo():

    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    Q = 1e-2*np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    R = 2.
    itr = 100

    real_state = []
    x = np.array([10, 5]).reshape(2, 1)

    for i in range(itr):
        real_state.append(x[0, 0])
        x = np.dot(F, x)+np.random.multivariate_normal(mean=(0, 0),
                                                       cov=Q).reshape(2, 1)

    measurements = [x+np.random.normal(0, R) for x in real_state]

    # initialization
    P = np.array([[10, 5], [5, 10]])
    x = np.random.multivariate_normal(mean=(10, 5), cov=P).reshape(2, 1)

    # filter
    # dim_x: hidden state size, dim_z: measurement size
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # Define parameters
    kf.x = x  # Initial state [position, speed]
    kf.F = F  # State transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement matrix
    kf.P = P  # Initial state covariance
    kf.R = R  # Measurement noise
    kf.Q = Q_discrete_white_noise(
        dim=2, dt=dt, var=1e-2)  # Process (system) noise

    filter_result = list()
    filter_result.append(x)
    for i in range(1, itr):
        z = measurements[i]
        kf.predict()
        kf.update(z)
        filter_result.append(kf.x)
    filter_result = np.squeeze(np.array(filter_result))

    return measurements, real_state, filter_result


def plot_result(measurements, real_state, filter_result):

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(measurements)),
             measurements[1:], label='Measurements')
    plt.plot(range(1, len(real_state)), real_state[1:], label='Real statement')
    plt.plot(range(1, len(filter_result)), np.array(
        filter_result)[1:, 0], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.axhline(5, label='Real statement')  # , label='$GT_x(real)$'
    plt.plot(range(1, len(filter_result)), np.array(
        filter_result)[1:, 1], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)
    plt.show()


if __name__ == '__main__':

    measurements, real_state, filter_result = demo()
    plot_result(measurements, real_state, filter_result)
