from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)


def detect():
    # 读取csv文件
    data = pd.read_csv("./data/postgraduate.csv", sep=",")
    # 转为np.array
    datalist = np.array(data)
    # print(datalist)
    # x
    x = datalist[:, 0]
    x = x.reshape(-1, 1)
    # y
    y = datalist[:, 1]
    # 绘制散点图
    # plt.scatter(x, y)
    # plt.show()
    # 多项式回归
    poly = PolynomialFeatures(degree=4)
    poly.fit(x)
    x_poly = poly.transform(x)
    # 线性回归
    reg = LinearRegression()
    reg.fit(x_poly, y)

    # 预测
    x_predict = poly.transform([[2025]])
    print(reg.predict(x_predict))

    X = np.arange(2000, 2025, 0.1).reshape(-1, 1)
    X_poly = poly.transform(X)
    Y = reg.predict(X_poly)

    plt.scatter(X, Y)
    plt.scatter(x, y)
    plt.savefig("predict.jpg")
    plt.show()


def main():
    detect()


if __name__ == "__main__":
    main()
