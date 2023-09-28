# machine2deeplearning_lab
##  工业数据的机器学习和深度学习算法预测

**Machine or Deep Learning for HEU Lab**

目录结构

#### 机器学习
算法库 **Sklearn**
- 线性回归
- 逻辑回归
- 决策树
- Adaboost-(分类任务)

#### 深度学习
依赖 **Pytorch**，框架 **Pytorch_Lightning**
- 分类(npv混凝土强度数据)
- 回归(鸢尾花分类数据)

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
