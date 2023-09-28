import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Net(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=5)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=5, out_features=out_dim)
        self.active2 = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        return x
    
class Class_LightningNet(pl.LightningModule):
    def __init__(self, indim, outdim) -> None:
        super().__init__()
        self.model = Net(indim, outdim)
        self.loss = nn.CrossEntropyLoss()
        self.val_step_loss = []
        
    def forward(self, data):
        result = self.model(data)
        return result
    
    def training_step(self, batch, batch_nb):
        data, target = batch
        data = data.to(torch.float32)
        target = target.long()
        output = self.forward(data)
        y_pred = torch.argmax(output, dim=1) # 取最大概率对应下标为预测类别
        loss = self.loss(output, target)
        self.log("train_step_loss", loss)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_nb):
        data, target = batch
        data = data.to(torch.float32)
        target = target.long()
        output = self.forward(data)
        y_pred = torch.argmax(output, dim=1)
        ce_loss = self.loss(output, target)
        self.log("validate_step_loss", ce_loss)
        self.val_step_loss.append(ce_loss)
    
    def on_validation_epoch_end(self) -> None:
        val_epoch_ce_loss = torch.stack(self.val_step_loss).mean()
        self.log('val_epoch_ce_loss', val_epoch_ce_loss)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        # 学习率自动调整策略
        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode='min', 
                                           factor=0.5, 
                                           patience=10, 
                                           verbose=True, 
                                           min_lr=1e-6, 
                                           threshold=0.0001)
        self.scheduler_dict = {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'train_step_loss',  # 监控的内容需要是self.log添加记录的内容
                'interval': 'epoch',   # 调整学习率的频率（每个epoch）
                'reduce_on_plateau': True,  # 使用 ReduceLROnPlateau 调度器
            }
        }
        # 如果想用学习率自动下降，可以返回scheduler_dict
        # return self.scheduler_dict
        return self.optimizer
        
    
        
    