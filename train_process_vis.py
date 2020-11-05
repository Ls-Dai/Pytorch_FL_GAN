import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils import parse

if __name__ == '__main__':

    num_of_clients = 0
    for _ in os.listdir("clients"):
        num_of_clients += 1

    file_path_list = []
    for i in range(num_of_clients):
        file_path = "clients/" + str(i) + "/log.csv"
        file_path_list.append(file_path)

    client_curve_list = []
    plt.figure('Loss of Gs')
    for count, file_path in enumerate(file_path_list):
        df = pd.read_csv(file_path, header=None)
        arr = np.array(df)
        lst = arr.tolist()
        loss_curve = []
        for values in lst:
            loss_curve.append(values[1])
        plt.plot(loss_curve)
        client_curve_list.append('client_' + str(count))
    plt.legend(client_curve_list)

    plt.figure('Loss of Ds')
    for count, file_path in enumerate(file_path_list):
        df = pd.read_csv(file_path, header=None)
        arr = np.array(df)
        lst = arr.tolist()
        loss_curve = []
        for values in lst:
            loss_curve.append(values[2])
        plt.plot(loss_curve)
        client_curve_list.append('client_' + str(count))
    plt.legend(client_curve_list)

    plt.show()
