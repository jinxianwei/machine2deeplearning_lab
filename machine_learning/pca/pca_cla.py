import os
import sys
current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_path)
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.read_csv_data import read_csv_data

def main():
    # 1. 读取csv数据
    name_dict, data = read_csv_data("dataset/iris.csv")
    # 鸢尾花数据集三类标签
    label_dict = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    label_list = ['setosa', 'versicolor', 'virginica']
    
    # 2. 确定特征和标签
    x = data[:, :-1]
    y = data[:, -1]
    
    # 3. 处理特征
    # 在主成分分析PCA之前，需要对特征进行标准化，确保所有特征在相同尺度下均衡
    x = StandardScaler().fit_transform(x)
    
    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test= train_test_split(x, y, stratify=y, test_size=0.5, random_state=0)
    # 在训练过程中，可用的只有训练集， 测试集的数据边换也需要根据训练集的数据进行变换
    x_t = StandardScaler().fit(x_train)
    x_train = x_t.transform(x_train)
    x_test = x_t.transform(x_test)
    
    # 对训练集和测试集分别进行PCA降维处理
    k = 0.98 # 设置降维占比
    pca = PCA(n_components=k)
    x_train_pca = pca.fit_transform(x_train) # 在训练集上拟合模型并进行降维
    x_test_pca = pca.transform(x_test) # 将测试集降维
    print("主成分的数量: {}".format(pca.n_components_))
    # 结果显示含义为：当维度降低到xx时，保留了原特征98%的信息   
    
    # 5. 利用降维后的训练集建立逻辑回归模型
    model = LogisticRegression()
    model.fit(x_train_pca, y_train)
    
    # 6. 对降维后的测试集进行分类，并进行模型评估
    y_pred = model.predict(x_test_pca)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    print(f"精确率为{accuracy}, 准确度为{precision}, 召回率为{recall}, F1分数为{f1}")
    
    report = classification_report(y_true=y_test, y_pred=y_pred)
    print(report)
    
    # 对降维后的前两个主成分进行类别的可视化
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], label_list):
        plt.scatter(
            x_train_pca[y_train == i, 0], x_train_pca[y_train == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("x_train_pca of IRIS dataset")
    plt.savefig("./x_train_pca.png")
    
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], label_list):
        plt.scatter(
            x_test_pca[y_test == i, 0], x_test_pca[y_test == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("x_test_pca of IRIS dataset")
    plt.savefig("./x_test_pca.png")
    

if __name__ == "__main__":
    main()