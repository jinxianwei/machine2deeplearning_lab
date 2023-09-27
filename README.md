# machine2deeplearning_lab
##  工业数据的机器学习和深度学习算法预测

**Machine or Deep Learning for HEU Lab**

目录结构

#### 机器学习
算法库 **Sklearn**
- 线性回归
- 逻辑回归
- 决策树

#### 深度学习
依赖 **Pytorch**，框架 **Pytorch_Lightning**
- 分类
- 回归

###### 注意
[X] 运行前，确保终端当前路径在项目根目录

[X] 深度学习任务，windows和linux在读取数据为dataloader时，需要注意num_workers参数的设置

运行命令
```bash
# 线性回归
python machine_learning/linear_regression/train.py

# 逻辑回归
python machine_learning/logistic_regression/train.py

# 深度回归
python  deep_learning/regression/train.py

## 深度学习 查看tensorboard损失曲线变化
tensorboard --logdir /home/bennie/bennie/temp/machine2deeplearning_lab/deep_learning/regression/tb_logs/npvproject/version_0
```