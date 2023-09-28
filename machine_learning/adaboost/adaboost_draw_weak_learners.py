import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)

from utils.read_csv_data import read_csv_data

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    
    # 4. 保存不同弱分类器数量时，测试集上评估指标
    weak_learns = [i for i in range(1, 40)]
    all_accuracy = []
    all_precision = []
    all_recall = []
    all_f1 = []
    

    
    # 5. 建立模型
    for i in tqdm(range(1, 40)):
        clf = AdaBoostClassifier(n_estimators=i)
        clf.fit(x_train, y_train)
    
        # 计算评估指标
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        # print(f"精确率为{accuracy}, 准确度为{precision}, 召回率为{recall}, F1分数为{f1}")
        all_accuracy.append(accuracy)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    
    # 6. 绘制曲线
    plt.figure(figsize=(8, 6))
    plt.plot(weak_learns, all_accuracy,label='accuracy', marker='o', linestyle='-', color='b', linewidth=1)
    plt.title('Adaboost weaker number to accuracy')
    plt.xlabel('Weaker learner number')
    plt.ylabel('Test accuracy')
    plt.grid(True)
    plt.savefig('./accuracy.png')
    
if __name__ == "__main__":
    main()