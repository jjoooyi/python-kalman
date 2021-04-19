# https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np

# 위치만 읽는 센서(dim_z)를 사용하여 위치와 속도(dim_x) 추적하는 필터
f = KalmanFilter(dim_x=2, dim_z=1)

# x - 상태 측정 벡터, 상태(위치, 속도)에 대한 초기값 지정
# 2차원 배열로 표현할 경우
f.x = np.array([[2.],   # 위치
                [0.]])  # 속도
# 1차원 배열로 표현할 경우
f.x = np.array([2., 0.])

# 상태 전이 행렬 F 정의, 시간 변화에 따른 상태 변화 야기 시키는 행렬
f.F = np.array([[1., 1.], [0., 1.]])

# 측정 기능 H 정의
f.H = np.array([[1., 0.]])

# 공분산 행렬 P 정의, P는 이미 np.eye(dim_x)가 포함되어 있기 때문에 불확실성만 곱하여 정의
f.P *= 1000.
# 같은 값
f.P = np.array([[1000., 0.],
                [0., 1000, ]])

# 측정 노이즈 행렬 R 할당, 2차원 행렬
f.R = 5
# 같은 값
f.R = np.array([[5.]])

# 프로세스 노이즈 행렬 Q 할당
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

# 표준 예측 / 업데이트 루프 수행
'''
이제 표준 예측 / 업데이트 루프를 수행하십시오.

some_condition_is_true 동안 :

z = get_sensor_reading()
f.predict()
f.update(z)

do_something_with_estimate (f.x)
절차 형식

이 모듈에는 칼만 필터링을 수행하는 독립형 기능도 포함되어 있습니다. 당신이 물체의 팬이 아니라면 이것을 사용하십시오.

예

while True:
    z, R = read_sensor()
    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
'''
