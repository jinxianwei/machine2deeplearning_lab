import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

from utils.read_csv_data import read_csv_data

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

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
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)
    
    # 4. 建立模型
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    
    # 5. 得到结果
    # 保存模型的权重文件
    file_name = "machine_learning/adaboost/adaboost_model.pkl"
    joblib.dump(clf, file_name)
    
    # 6. 计算评估指标
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    print(f"精确率为{accuracy}, 准确度为{precision}, 召回率为{recall}, F1分数为{f1}")
    
    # 7. 加载保存的模型进行预测
    loaded_clf = joblib.load(file_name)
    predictions = loaded_clf.predict(x_test)
    print(f"加载保存的模型进行预测的结果为{predictions}")
if __name__ == "__main__":
    main()