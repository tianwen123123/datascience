import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# 数据集
"""
1.GRE Score: GRE成绩（满分340分)
2.TOEFL Score: 托福成绩（满分120分）
3.Unversity Rating: 大学评分（满分5分) 与 学分绩、绩点 相关
4.SOP: 目的陈述
5.LOR: 推荐信强度（满分5分）
6.CGPA: 本科GPA（满分10分）
7.Research: 研究经历（0或1）
8.Chance of Admin: 承认的机会（从0到1）
"""


# 展示信息
def show_info(data):
    # 查看 data 中的数据
    print(data.head())
    # 生成描述性统计数据
    print(data.describe())
    # 查看data数据中是否有缺失数据
    print(data.info())
    # 取出相关字段，生成相关性的可视化图
    field = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research', 'Chance of Admit']
    sns.heatmap(data[field].corr(), annot=True, vmin=-1, vmax=1)
    # 保存图像
    plt.savefig("./img/param_relation.jpg")
    plt.show()


# 预测
def predict(data):
    # x(特征),y(标签)
    x = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
    # print(x)
    y = data[['Chance of Admit']]
    # print(y)
    # 归一化
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    # print(x)
    # PCA降维
    pca = PCA(n_components=4)
    x = pca.fit_transform(x)
    print("=" * 20)
    print(pca.explained_variance_ratio_)
    print("=" * 20)
    print(x)
    # print(x.shape)
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)
    # 建立线性回归模型
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    y_pred = lm.predict(x_test)
    score = r2_score(y_test, y_pred)
    print(score)

    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend()
    plt.savefig("predict_success" + str(round(score, 2)) + ".jpg")
    plt.show()


def main():
    # 读csv文件
    data = pd.read_csv('./data/Admission_Predict.csv')
    # 展示信息
    show_info(data)
    # 预测
    #predict(data)


if __name__ == "__main__":
    main()
