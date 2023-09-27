from lightning_data_module import Npv_DataModule
from fully_connected_net import LightningNet
from torchvision import transforms
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


model = LightningNet(indim=8, outdim=1)
my_transform = transforms.Compose(
    [transforms.Lambda(lambda x: (x - np.mean(x)) / np.std(x))]
)
data_module = Npv_DataModule(data_path='/home/bennie/bennie/temp/machine2deeplearning_lab/dataset/npvproject-concrete.csv',
                             split_test=0.2,
                             batch_size=20,
                             num_workers=4,
                             transform=my_transform)
tensorboard_logger = TensorBoardLogger('tb_logs', name='npvproject')
trainer = Trainer(logger=tensorboard_logger,
                #   accelerator='gpu',
                #   devices=1,
                  max_epochs=20,
                  log_every_n_steps=1)
trainer.fit(model=model, datamodule=data_module)
