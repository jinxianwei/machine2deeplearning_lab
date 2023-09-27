import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl


class Net(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer1 = nn.Linear(in_features=in_dim, out_features=5)
        self.active1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=5, out_features=10)
        self.active2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=10, out_features=8)
        self.active3 = nn.ReLU()
        self.layer4 = nn.Linear(in_features=8, out_features=out_dim)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.active1(x)
        x = self.layer2(x)
        x = self.active2(x)
        x = self.layer3(x)
        x = self.active3(x)
        x = self.layer4(x)
        return x
    
class LightningNet(pl.LightningModule):
    def __init__(self, indim, outdim) -> None:
        super().__init__()
        self.model = Net(indim, outdim)
        self.loss = nn.MSELoss()
        self.val_step_loss = []
        
    def forward(self, data):
        result = self.model(data)
        return result
    
    def training_step(self, batch, batch_nb):
        data, target = batch
        data = data.to(torch.float32)
        target = target.to(torch.float32)
        output = self.forward(data)
        loss = self.loss(output, target)
        self.log("train_step_loss", loss)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_nb):
        data, target = batch
        data = data.to(torch.float32)
        target = target.to(torch.float32)
        output = self.forward(data)
        mse_loss = self.loss(output, target)
        self.log("validate_step_loss", mse_loss)
        self.val_step_loss.append(mse_loss)
    
    def on_validation_epoch_end(self) -> None:
        val_epoch_mse_loss = torch.stack(self.val_step_loss).mean()
        self.log('val_epoch_mse_loss', val_epoch_mse_loss)
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.001)
        
    
        
    