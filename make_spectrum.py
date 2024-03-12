import numpy as np
import pickle
import time
from datetime import datetime, timedelta
import pytz
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def make_transfer_matrix(n1, l1, wavelength, n2, l2):
    R = (n1 - n2) / (n1 + n2)
    T = 2 * n1 / (n1 + n2)
    delta = (2 * np.pi * n2 * l2) / wavelength
    j = 1j
    boundary = np.array([[1, R], [R, 1]]) / T
    if l2 == 0:
        propagation = [[1, 0], [0, 1]]
    else:
        propagation = np.array(
            [[np.exp(-delta * j), 0], [0, np.exp(delta * j)]])

    return boundary @ propagation


def matrix_mul(lst):
    a = [[1, 0], [0, 1]]
    for b in lst:
        a = a@b
    return a


def main():

    n_a = np.random.uniform(1, 2)  # 물체 a의 굴절률
    n_b = np.random.uniform(1, 2)  # 물체 b의 굴절률
    w_a = np.random.uniform(1, 500)
    w_b = np.random.uniform(1, 500)
    mat_list = [
        # 공기
        [1, 0],
        [n_a, w_a],
        [n_b, w_b],
        [n_a, w_a],
        [n_b, w_b],
        [n_a, w_a],
        [n_b, w_b],
        [n_a, w_a],
        [n_b, w_b],
        [1, 0]
    ]
    # 파장(wavelength) 범위 정의
    # 파장, 두께 단위 통일
    wave_values = np.linspace(400, 700, num=201)

    if os.path.exists("data.pkl"):
        with open("data.pkl", "rb") as file:
            data = pickle.load(file)
    else:
        data = []

    if os.path.exists("graph.pkl"):
        with open("graph.pkl", "rb") as file:
            graph = pickle.load(file)
    else:
        graph = []

    M_list = []  # 전달 행렬을 저장할 리스트
    result = []  # 전달 행렬 곱한 뒤 파장마다 저장할 리스트
    # 투과 계산
    for wave in wave_values:
        for i in range(len(mat_list) - 1):
            n_1, w_1 = mat_list[i]
            n_2, w_2 = mat_list[i + 1]
            M = make_transfer_matrix(n_1, w_1, wave, n_2, w_2)
            # You might want to do something with M here, or you can remove this line if it's not needed.
            M_list.append(M)
        result.append(matrix_mul(M_list))
        M_list.clear()

    # 전달 행렬 요소 추출
    M_elements = [M[0, 0] for M in result]
    M_elements_filtered1 = [1/x if x != 0 else 0 for x in M_elements]
    M_elements_filtered2 = [
        (abs(x))**2 if x != 0 else 0 for x in M_elements_filtered1]

    # 반사 계산, 점검용으로 사용
    # M_reflect = [M[1, 0]/M[0, 0] for M in result]
    # M_reflect_filtered1 = [(abs(x))**2 if x != 0 else 0 for x in M_reflect]
    # Transmittance + Reflectance = 1
    # M_add = [x+y for x, y in zip(M_elements_filtered2, M_reflect_filtered1)]

    # 그래프로 표시
    plt.figure(figsize=(15, 6))
    # plt.plot(wave_values * 1e9, M_reflect_filtered1,color='blue', label='Transmission')
    plt.plot(wave_values, M_elements_filtered2,
             color='green', label='Transmission')
    plt.xlabel('wavelength')
    plt.ylabel('transmission')
    plt.title('Transmission')
    # plt.legend()
    plt.grid(True)
    # plt.show()

    # 변수 및 그래프 데이터 저장
    new_data = [n_a, w_a, n_b, w_b]
    data.append(new_data)
    new_graph = M_elements_filtered2
    graph.append(new_graph)
    with open("data.pkl", "wb") as file:
        pickle.dump(data, file)
    with open('graph.pkl', 'wb') as file:
        pickle.dump(graph, file)

    plt.close()


if __name__ == "__main__":
    # 현재 시간을 로컬 시간대로 가져옴 (예: 서울 시간대)
    local_timezone = pytz.timezone("Asia/Seoul")
    end_time = datetime.now(local_timezone) + timedelta(hours=1)  # hours=1

    # 시간을 hh-mm-ss 포맷으로 변환
    print("현재 시간:", datetime.now(local_timezone).strftime('%Y-%m-%d %H:%M:%S'))
    print("종료예정시간 : ", end_time.strftime('%Y-%m-%d %H:%M:%S'))

    while datetime.now(local_timezone) < end_time:

        main()  # main 함수 실행
        time.sleep(1)  # 1초 대기
