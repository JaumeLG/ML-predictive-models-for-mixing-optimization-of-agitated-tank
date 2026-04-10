import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset
from torchmetrics.image import StructuralSimilarityIndexMeasure

import numpy as np
import matplotlib.pyplot as plt
from math import exp
from sklearn.preprocessing import StandardScaler

from time import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valSplit=0.8
Epochs= 5000 #En el model final parava a les 2000. Mirar en el early stopping quan para i comparar predictive errors.
LR=1e-4
batch=32

# Basic setup for early stopping criteria
patience = 500  # epochs to wait after no improvement
delta = 0.005  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 500  # count of epochs with no improvement

print('Loading data')
inputs=np.load('./dataset/inputsMLP.npz')['arr_0']
outputs=np.load('./dataset/outputs.npz')['arr_0']

print('Scaling data')
scalerIn=StandardScaler()
inputs_sc=scalerIn.fit_transform(inputs)

outMean=np.mean(outputs)
outStd=np.std(outputs)
outputs_sc=(outputs-outMean)/outStd

print('Creating dataset')
train_size = int(inputs.shape[0]*valSplit)
inputs_sc=torch.Tensor(inputs_sc)
outputs_sc=torch.Tensor(outputs_sc)

dataset=TensorDataset(inputs_sc,outputs_sc)

val_size = inputs.shape[0] - train_size -320
    
usableIndexes = list(range(len(dataset)-320))
usableSet = Subset(dataset,usableIndexes)
train_dataset, test_dataset = random_split(usableSet, [train_size, val_size],generator=torch.Generator().manual_seed(37))

train_DL = DataLoader(train_dataset,batch_size=batch,shuffle=True,num_workers=4,pin_memory=True)
test_DL = DataLoader(test_dataset,batch_size=8,shuffle=False,num_workers=2,pin_memory=True)

valIndexes = list(range(len(dataset)-320,len(dataset)))
val_dataset=Subset(dataset,valIndexes)
val_DL = DataLoader(val_dataset,batch_size=8,shuffle=False)

class hybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1=nn.Linear(inputs.shape[1],32)
        self.fc2=nn.Linear(32,64)
        self.fc3=nn.Linear(64,128)
        self.fc4=nn.Linear(128,216)
        self.fc5=nn.Linear(216,648)

        self.cnv1=nn.Conv2d(3,512,3,1,padding=1)
        self.up1=nn.ConvTranspose2d(512,512,3,(2,2),padding=1)
        self.cnv2=nn.Conv2d(512,256,3,1,padding=1)
        self.up2=nn.ConvTranspose2d(256,256,3,(2,2))
        self.cnv3=nn.Conv2d(256,128,3,1,padding=1,padding_mode='replicate')
        self.cnv4=nn.Conv2d(128,64,3,1,padding=1,padding_mode='replicate')
        self.cnv5=nn.Conv2d(64,32,3,1,padding=1,padding_mode='replicate')
        self.cnv6=nn.Conv2d(32,3,1,1)


    def forward(self,x):
        x=F.elu(self.fc1(x))
        x=F.elu(self.fc2(x))
        x=F.elu(self.fc3(x))
        x=F.elu(self.fc4(x))
        x=F.elu(self.fc5(x))

        x=x.view(-1,3,18,12)

        x=F.elu(self.cnv1(x))
        x=self.up1(x)

        x=F.elu(self.cnv2(x))
        x=self.up2(x)
        x=F.pad(x,[0,1,0,1])

        x=F.elu(self.cnv3(x))
        x=F.elu(self.cnv4(x))
        x=F.elu(self.cnv5(x))
        x=self.cnv6(x)

        return x
    
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        

model = hybridModel()
model.apply(init_weights)
model = model.to(device)

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
torch.save(model.state_dict(), './results/models/model_hybrid_2d.pth')

print('Saving traning curves')
plt.figure(figsize=(10, 6))
plt.plot(torch.Tensor(train_loss), label='Training Loss')
plt.plot(torch.Tensor(test_loss), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('./results/plots/training_losses_hybrid_2d.png') 
plt.close()


print('Evaluating model')
model.eval()

with torch.no_grad():
    for k,(x_test,y_test) in enumerate(val_DL):
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
np.savez('./results/preds/preds_hybrid_2d.npz',ypred_sc)
np.savez('./results/preds/true_2d.npz',ytrue_sc)

pred_mod=np.sqrt(np.sum(ypred_sc**2,axis=1))
true_mod=np.sqrt(np.sum(ytrue_sc**2,axis=1))

print(pred_mod.shape)

deadVolume=np.zeros((10,2))

for i in range(10):
    pred_dv=(pred_mod[i*32:(i+1)*32]<0.01).sum()/(32*pred_mod[0].size)*100
    true_dv=(true_mod[i*32:(i+1)*32]<0.01).sum()/(32*pred_mod[0].size)*100

    deadVolume[i,0]=pred_dv
    deadVolume[i,1]=true_dv

errorVolume = (deadVolume[:,1]-deadVolume[:,0])

print(deadVolume)

print(np.mean(np.abs(errorVolume)),np.max(np.abs(errorVolume)))
