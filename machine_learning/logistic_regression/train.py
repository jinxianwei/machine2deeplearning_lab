import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

from utils.read_csv_data import read_csv_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")
    print(name_dict, data.shape)
    # 鸢尾花数据集三类标签
    label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    
    # 2. 确定特征和标签
    x = data[:, 0:-1]
    y = data[:, -1]
    
    # 3. 划分训练集和测试集
    # TODO 数据集是否划分取决于模型，线性回归模型没有划分数据集的必要
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.5, random_state=0)
    
    # 4. 建立模型训练
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    
    # 5. 得到结果
    print("系数 {}".format(classifier.coef_))
    print("截距 {}".format(classifier.intercept_))
    
    # 6. 计算评估指标
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    print(f"精确率为{accuracy}, 准确度为{precision}, 召回率为{recall}, F1分数为{f1}")
    

if __name__ == "__main__":
    main()