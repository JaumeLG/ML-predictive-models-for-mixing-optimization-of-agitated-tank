import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
import matplotlib.pyplot as plt
from math import exp

from time import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valSplit=0.8
Epochs= 5000 
LR=1e-4
batch=32

# Basic setup for early stopping criteria
patience = 500  # epochs to wait after no improvement
delta = 0.005  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 500  # count of epochs with no improvement


print('Loading data')
inputs=np.load('./dataset/inputsCNN.npz')['arr_0']
outputs=np.load('./dataset/outputs3D.npz')['arr_0']

print('Scaling data')
inMean=np.mean(inputs)
inStd=np.std(inputs)
inputs_sc=(inputs-inMean)/inStd

outMean=np.mean(outputs)
outStd=np.std(outputs)
outputs_sc=(outputs-outMean)/outStd

print('Creating dataset')
train_size = int(inputs.shape[0]*valSplit)
inputs_sc=torch.Tensor(inputs_sc)
outputs_sc=torch.Tensor(outputs_sc)

dataset=TensorDataset(inputs_sc,outputs_sc)

val_size = inputs.shape[0] - train_size -10
    
usableIndexes = list(range(len(dataset)-10))
usableSet = Subset(dataset,usableIndexes)
train_dataset, test_dataset = random_split(usableSet, [train_size, val_size],generator=torch.Generator().manual_seed(37))
    
train_DL = DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=4,pin_memory=True)
test_DL = DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=2,pin_memory=True)

valIndexes = list(range(len(dataset)-10,len(dataset)))
val_dataset=Subset(dataset,valIndexes)
val_DL = DataLoader(val_dataset,batch_size=8,shuffle=False)

print('Defining CNN model')
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=1.0,inplace=True),#inplace=True
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1,padding_mode='reflect'),
            nn.BatchNorm3d(out_channels),
            nn.ELU(alpha=1.0,inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(5, 32)        # 6 -> 32
        self.enc2 = DoubleConv(32, 64)    # 32 -> 64
        self.enc3 = DoubleConv(64, 128)  # 64 -> 128
        self.enc4 = DoubleConv(128, 256)  # 128 -> 256
        
        # Decoder with output padding to match encoder dimensions
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)#, output_padding=(1,1,1)
        self.dec3 = DoubleConv(256, 128)  # (256 + 128) -> 128
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) #, output_padding=(1,1,0)
        self.dec2 = DoubleConv(128, 64)  # (128 + 64) -> 64
        
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)    # (64 + 32) -> 32
        
        # Final convolution
        self.oneLast_conv = nn.Conv3d(32, 3, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder path
        dec3 = self.up3(enc4)
        if dec3.size() != enc3.size():
            diff_y = enc3.size(2) - dec3.size(2)
            diff_x = enc3.size(3) - dec3.size(3)
            dec3 = nn.functional.pad(dec3, (diff_x//2, diff_x - diff_x//2,
                                          diff_y//2, diff_y - diff_y//2))
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        if dec2.size() != enc2.size():
            diff_y = enc2.size(2) - dec2.size(2)
            diff_x = enc2.size(3) - dec2.size(3)
            dec2 = nn.functional.pad(dec2, (diff_x//2, diff_x - diff_x//2,
                                          diff_y//2, diff_y - diff_y//2))
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        if dec1.size() != enc1.size():
            diff_y = enc1.size(2) - dec1.size(2)
            diff_x = enc1.size(3) - dec1.size(3)
            dec1 = nn.functional.pad(dec1, (diff_x//2, diff_x - diff_x//2,
                                          diff_y//2, diff_y - diff_y//2))
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        

        return self.oneLast_conv(dec1)
    
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight)

model = UNet3D()
model = model.to(device)
model.apply(init_weights)

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

SSIM_WEIGHT=0.8
MSE_WEIGHT=0.2

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        ssim_loss = 1 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        total_loss = SSIM_WEIGHT * ssim_loss + MSE_WEIGHT * mse_loss
        return total_loss

lossF = CustomLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=LR)#4.69e-4

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

# Initialize early stopping
early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)


print('Starting training')
epochs = Epochs
train_loss = []
test_loss = []

start=time()
for i in range(epochs):
    avgTrainLoss = 0
    idx = 0
    
    model.train()

    epoch_start=time()
    for (x_train,y_train) in train_DL:
        optimizer.zero_grad()
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        
        y_pred = model(x_train)
        batchLoss = lossF(y_pred,y_train)
        batchLoss.backward()
        optimizer.step()
        
        avgTrainLoss += batchLoss.item()

    avgTrainLoss /= len(train_DL)
    train_loss.append(avgTrainLoss)  
        
    model.eval()
    with torch.no_grad():
        avgTestLoss=0
        for (x_test,y_test) in test_DL:
            x_test=x_test.to(device)
            y_test=y_test.to(device)
            ypred_val=model(x_test)
            testLoss = lossF(ypred_val, y_test)
            avgTestLoss += testLoss.item()
                
        avgTestLoss /= len(test_DL)
        test_loss.append(avgTestLoss)

    # Check early stopping condition
    early_stopping.check_early_stop(avgTestLoss)
    
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {i}")
        break

    print(f'Epoch: {i}, train loss: {train_loss[i]}, test loss: {test_loss[i]}, time per epoch: {round(time()-epoch_start,3)}')
          
total_time=time()-start
print(f'Total training time: {total_time/60}')

print('Saving model')
torch.save(model.state_dict(), './results/models/model_Unet_3d.pth')

print('Saving traning curves')
plt.figure(figsize=(10, 6))
plt.plot(torch.Tensor(train_loss), label='Training Loss')
plt.plot(torch.Tensor(test_loss), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('./results/plots/training_losses_Unet_3d.png')
plt.close()


print('Evaluating model')
model.eval()

with torch.no_grad():
    for k,(x_test,y_test) in enumerate(test_DL):
        x_test=x_test.to(device)
        y_test=y_test.to(device)
        if k==0:
            ypred_val=model(x_test)
            ypred_val=ypred_val
            ytest=y_test
        else:
            temp=model(x_test)
            temp=temp
            ypred_val=torch.cat([ypred_val,temp])
            ytest=torch.cat([ytest,y_test])
        
ypred_val=ypred_val.to('cpu')
ytest=ytest.to('cpu')
ypred_val=np.array(ypred_val)
ytest=np.array(ytest)

error0=np.mean(abs(ypred_val-ytest))
error1=np.mean(abs(ypred_val-ytest)/(ytest+1e-5))
error2=np.mean(abs(ypred_val-ytest)/np.std(ytest,axis=0))
error3=np.mean(abs(ypred_val-ytest)/np.max(ytest,axis=0))

print(f'MAE: {error0}, RelErr: {error1}, RelErr_std: {error2}, RelErr_inf: {error3}')


print('Scaling predictions')

ypred_sc=ypred_val*outStd+outMean
ytrue_sc=ytest*outStd+outMean

print('Shape preds:',ypred_sc.shape)
np.savez('./results/preds/preds_Unet_3d.npz',ypred_sc)

pred_mod=np.sqrt(np.sum(ypred_sc**2,axis=1))
true_mod=np.sqrt(np.sum(ytrue_sc**2,axis=1))

print(pred_mod.shape)

deadVolume=np.zeros((pred_mod.shape[0],2))

for i in range(pred_mod.shape[0]):
    pred_dv=(pred_mod[i]<0.01).sum()/(pred_mod[0].size)*100
    true_dv=(true_mod[i]<0.01).sum()/(pred_mod[0].size)*100

    deadVolume[i,0]=pred_dv
    deadVolume[i,1]=true_dv

errorVolume = (deadVolume[:,1]-deadVolume[:,0])

print(deadVolume)

print(np.mean(np.abs(errorVolume)),np.max(np.abs(errorVolume)))