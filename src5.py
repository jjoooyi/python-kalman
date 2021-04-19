import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver

r_std, q_std = 2., 0.003
f = KalmanFilter(dim_x=2, dim_z=1)
f.x = np.array([0., 1.])  # 위치, 속도
f.F = np.array([[1., 1.],
                [0., 1.]])
f.R = np.array([[r_std**2]])
f.H = np.array([[1., 0.]])
f.P = np.diag([.1**2, .03**2])
f.Q = Q_discrete_white_noise(2, 1., q_std**2)

saver = Saver(f)
for z in range(100):
    f.predict()
    f.update([z + np.random.randn() * r_std])
    saver.save()  # save the filter's state

saver.to_array()

print(saver)

plt.figure()
# plt.plot(saver.x[:, :])
plt.plot(saver.x_prior[:, :])
# 마할라노비스 거리(Mahalanobis distance): 변수의 분산과 상관성을 고려한 거리 측정 방법
# 변수 간의 상관 관계가 있을 때 유용하게 활용할 수 있는 방법
plt.plot(saver.mahalanobis)
plt.show()


# x = np.array([[0., 1.]])  # position, velocity
# xs = []
# F = np.array([[1, 1.], [0, 1]])
# R = np.array([[r_std**2]])
# H = np.array([[1., 0.]])
# P = np.diag([.1**2, .03**2])
# Q = Q_discrete_white_noise(2, 1., q_std**2)
# for z in range(100):
#     x, P = f.predict(x, P, F=F, Q=Q)
#     x, P = f.update(x, P, z=[z + np.random.randn() * r_std], R=R, H=H)
#     xs.append(x[0, 0])
# plt.plot(xs)
# plt.show()
