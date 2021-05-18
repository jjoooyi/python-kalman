import numpy as np
import random
import matplotlib.pyplot as plt


def Hjacob(xp):
    """
    input: xp(추정값)
    output: H(출력행렬)
    측정모델의 출력행렬 H 를 생성한다. 측정모델은 비선형이기 때문에 선형으로 근사시킨 행렬 H 가 필요하다.
    """
    H = np.zeros((1, 3))

    x1 = xp[0, 0]
    x3 = xp[2, 0]

    H[0, 0] = x1 / np.sqrt(x1**2 + x3**2)
    H[0, 1] = 0
    H[0, 2] = x3 / np.sqrt(x1**2 + x3**2)

    return H

#Hjacob(np.array([0, 90, 1100]).reshape(-1, 1))


def hx(xhat):
    """
    input: xhat[0,0]-위치, xhat[1,0]-속도, xhat[2,0]-고도
    output: zp(목표물까지의 거리)
    직선 거리 변환
    """
    x1 = xhat[0, 0]  # 위치
    x3 = xhat[2, 0]  # 고도

    zp = np.sqrt(x1**2 + x3**2)  # 거리 = sqrt(위치^2 + 고도^2)

    return zp

#hx(np.array([3, 2, 4]).reshape(-1, 1))


dt = 0.05
A = np.eye(3) + dt * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # 시스템 행렬 3x3


Q = np.array([[0, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])  # 시스템 오차 3x3
R = np.array([10]).reshape(-1, 1)  # 측정 오차 1x1

# XXX: 측정 모델의 행렬은 비선형함수에서 선형으로 근사화된 행렬로 구성한다.
x = np.array([0, 90, 1100]).reshape(-1, 1)  # 임의로 예측한 초기 추정값 3x1

P = np.eye(3) * 10  # 초기 추정값에 대한 오차 공분산 3x3


def RadarEKF(z, dt):
    """
    input: z(측정 거리), dt(샘플링 간격)
    output: 수평 위치, 이동 속도, 고도
    """

    global x, P, A, Q, R

    # A = np.eye(3) + dt * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) # 시스템 행렬 3x3

    H = Hjacob(x)  # 출력 행렬 1x3, 측정모델(비선형)에 대한 선형근사화를 구한다.

    # 상태변수 추정값과 오차 공분산 예측
    xp = A.dot(x)  # 3x3 * 3x1 = 3x1
    Pp = A.dot(P).dot(A.T) + Q  # 오차공분산에 시스템 오차 추가 3x3 * 3*3 * 3x3 + 3x3 = 3x3
    # print(A)

    # 칼만 이득 계산
    # 3x3 * 3x1 * (1x3 * 3x3 * 3x1 + 1x1) = 3x1
    K = Pp.dot(H.T).dot(np.linalg.inv(H.dot(Pp).dot(H.T) + R))
    # print(K)

    # 추정값 계산
    # x = xp + K.dot(z - hx(xp)) # 3x1 + 3x1 * (scalar - scalar) = 3x1
    # 3x1 + 3x1 * (scalar - scalar) = 3x1, 비선형함수 그대로 사용함
    x = xp + K * (z - hx(xp))

    # 오차 공분산 계산
    P = Pp - K.dot(H).dot(Pp)  # 3x3 - 3x1 * 1x3 * 3x3 = 3x3

    pos = x[0, 0]
    vel = x[1, 0]
    alt = x[2, 0]

    return pos, vel, alt

#RadarEKF(1002.9, 0.05)


posp = 0  # 이전 측정값의 위치


def GetRadar(dt):
    """
    input:
    output:

    """
    global posp

    vel = 100 + 5 * np.random.normal()  # 이동거리에 대한 랜덤값 추가
    alt = 1000 + 10 * np.random.normal()  # 이동거리에 대한 랜덤값 추가

    pos = posp + vel*dt  # 위치 = 이전 위치 + 속도 * 이동시간

    v = 0 + pos * 0.05 * np.random.normal()  # 측정 오차를 일부러 추가함
    r = np.sqrt(pos**2 + alt**2) + v

    posp = pos  # 이전 위치값을 계산된 위치값으로 갱신
    # print(posp)

    return r

# GetRadar(0.05)


def TestRadarEKF():
    """
    테스트 프로그램
    """
    dt = 0.05  # 0.05초 샘플링 간격
    t = np.arange(0, 20, dt)  # 0~20까지 0.05초 간격으로 샘플링

    z_saved = []  # 측정값
    x_saved = []  # 추정값

    for i in t:
        r = GetRadar(dt)  # 측정값을 구한다.
        pos, vel, alt = RadarEKF(r, dt)  # 추정값을 구한다.

        z_saved.append(r)  # 측정값 저장
        x_saved.append([pos, vel, alt])  # 추정값 저장

    PosSaved = [item[0] for item in x_saved]
    VelSaved = [item[1] for item in x_saved]
    AltSaved = [item[2] for item in x_saved]

    # [그림 12-3]
    fig, ax = plt.subplots(4, 1, figsize=(12, 8))
    ax[0].plot(t, PosSaved, c='r', label='estimate Position')  # 측정값
    ax[0].legend()
    ax[1].plot(t, VelSaved, c='b', label='estimate Velocity')  # 측정값
    ax[1].legend()
    ax[2].plot(t, AltSaved, c='g', label='estimate Altitude')  # 측정값
    ax[2].legend()
    ax[3].plot(t, z_saved, c='y', label='measured pos')
    ax[3].legend()
    plt.show()


TestRadarEKF()
