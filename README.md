# machine2deeplearning_lab
##  工业数据的机器学习和深度学习算法预测

**Machine or Deep Learning for HEU Lab**

![image](https://github.com/jinxianwei/CloudImg/assets/81373517/c08e2c56-179c-4568-8eeb-ae0047103c04)
![image](https://github.com/jinxianwei/CloudImg/assets/81373517/ddbe8b3e-02a5-42a9-8bb9-d53cb3792ce0)
![image](https://github.com/jinxianwei/CloudImg/assets/81373517/c452fb0b-675a-4ff0-b537-04af7506b1fe)

目录结构

#### 机器学习
算法库 **Sklearn**
- [x] 线性回归
- [x] 逻辑回归
- [ ] 决策树
- [x] Adaboost-(分类任务)
不同数量弱分类器在测试集上的accuracy
![accuracy](https://github.com/jinxianwei/CloudImg/assets/81373517/3c4c7afa-2e5e-4679-8852-81302ea6045a)
- [x] Nearest Neighbors -(分类任务)
不同Neighbors权重类型模型在测试集决策边界的可视化
![k_neighbors](https://github.com/jinxianwei/CloudImg/assets/81373517/4b25b680-c883-48e2-9846-357959fe7363)
- [x] SVM-(分类任务)
不同SVM分类器在测试集决策边界的可视化
![svm](https://github.com/jinxianwei/CloudImg/assets/81373517/2a154234-ba2a-45d8-88ef-0ea4bd59cabf)
- [x] LogisticRegression with PCA
训练集和测试集特征前两个主成分在类别上的可视化
![x_train_pca](https://github.com/jinxianwei/CloudImg/assets/81373517/00878756-df1f-4e64-a04b-213371fda10b)
![x_test_pca](https://github.com/jinxianwei/CloudImg/assets/81373517/d14fa1de-e5bf-46f2-8707-91d86bb2be21)
#### 深度学习
依赖 **Pytorch**，框架 **Pytorch_Lightning**
- [x] 回归任务(npv混凝土强度数据)
- [x] 分类任务(鸢尾花分类数据)

###### 注意
- [x]  运行前，确保终端当前路径在项目根目录

- [x]  深度学习任务，windows和linux在读取数据为dataloader时，需要注意num_workers参数的设置

运行命令
```bash
# 线性回归
python machine_learning/linear_regression/train.py

# 逻辑回归
python machine_learning/logistic_regression/train.py

# 逐步增强法(Adaboost)(分类任务)
python machine_learning/adaboost/adaboost_classifier.py

# 深度回归
python  deep_learning/regression/train.py

# 深度分类
python deep_learning/classification/train.py

## 深度学习 查看tensorboard损失曲线变化
tensorboard --logdir tb_logs/npvproject/version_0
```
