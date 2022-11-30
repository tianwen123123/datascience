import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

np.set_printoptions(suppress=True)


def cal():
    data = pd.read_csv("./data/postgraduate.csv", sep=",")
    datalist = np.array(data)
    for i in range(1, len(datalist)):
        # 计算增长数量
        datalist[i][2] = round(datalist[i][1] - datalist[i - 1][1], 2)
        # 计算增长率
        datalist[i][3] = round((datalist[i][2] * 1.0) / datalist[i - 1][1] * 100, 2)
    # 输出结果
    print(datalist)
    # 写回csv
    new_data = pd.DataFrame(datalist, columns=['YEAR', 'PGN', 'INC', 'GR'], dtype=float)
    new_data.to_csv("./data/postgraduate.csv", sep=",", index=False)


def paint():
    # 第一，读取数据
    df = pd.read_csv("./data/postgraduate.csv", encoding='utf-8')
    print(df)
    # 第二，绘制折线图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 可以解释中文无法显示的问题
    # 1)创建画布
    plt.figure(figsize=(10, 5), dpi=100)
    # 2)绘制图像
    # bmh、ggplot、dark_background、fivethirtyeight和grayscale
    plt.style.use('ggplot')
    # 横纵点
    plt.bar(df["YEAR"], df["PGN"], label="报名人数")
    plt.title("近年考研报名情况")
    plt.xlabel("年份")
    plt.ylabel("报名数量(单位:万人)")
    # 设置数字标签
    for a, b in zip(df["YEAR"], df["PGN"]):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.grid(True)
    # 保存图像
    plt.savefig("PGN.jpg")
    # 3)展示图像
    plt.show()

def main():
    # cal()
    paint()


if __name__ == "__main__":
    main()
