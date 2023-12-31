from lightning_data_module import IRIS_DataModule
from fully_connected_net import Class_LightningNet

from torchvision import transforms
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# iris数据集，4个特征，3种类别
model = Class_LightningNet(indim=4, outdim=3)
# 对数据进行归一化，利于模型更快收敛
my_transform = transforms.Compose(
    [transforms.Lambda(lambda x: (x - np.mean(x)) / np.std(x))]
)
# TODO windows 需要删除num_workers参数，不支持多进程读取数据
data_module = IRIS_DataModule(data_path='dataset/iris.csv',
                             split_test=0.2,
                             batch_size=20,
                            #  num_workers=4,
                             transform=my_transform)
tensorboard_logger = TensorBoardLogger('tb_logs', name='irisproject')
trainer = Trainer(logger=tensorboard_logger,
                #   accelerator='gpu',
                #   devices=1,
                  max_epochs=1000,
                  log_every_n_steps=1)
trainer.fit(model=model, datamodule=data_module)
