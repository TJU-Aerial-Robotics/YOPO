import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_path = "/home/lu/flightmare/flightmare/run/utils/dist.csv"
    temp = np.loadtxt(file_path, dtype=float, delimiter=",")
    file_path = "/home/lu/flightmare/flightmare/run/utils/dist_x.csv"
    tempX = np.loadtxt(file_path, dtype=float, delimiter=",")
    plt.plot(tempX, temp)
    plt.show()
    print("dist min:", np.min(temp))
    file_path = "/home/lu/flightmare/flightmare/run/utils/ctrl_log.csv"
    ctrl_log = np.loadtxt(file_path, dtype=float, delimiter=",")
    v_total = np.sqrt(
        ctrl_log[:, 3] * ctrl_log[:, 3] + ctrl_log[:, 4] * ctrl_log[:, 4] + ctrl_log[:, 5] * ctrl_log[:, 5])
    print("v max: ", np.max(v_total))
    plt.plot(ctrl_log[:, 3], label='vx')
    plt.plot(ctrl_log[:, 4], label='vy')
    plt.plot(ctrl_log[:, 5], label='vz')
    plt.plot(v_total, label='v_total')
    plt.plot(ctrl_log[:, 6], label='ax')
    plt.plot(ctrl_log[:, 7], label='ay')
    plt.plot(ctrl_log[:, 8], label='az')
    plt.legend()
    plt.show()
