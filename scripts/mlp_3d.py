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
inputs=np.load('./dataset/inputs3D.npz')['arr_0']
outputs=np.load('./dataset/outputs3D.npz')['arr_0']

print(inputs.shape,outputs.shape)

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

class mlp(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1=nn.Linear(inputs.shape[1],50)
        self.fc2=nn.Linear(50,150)
        self.fc3=nn.Linear(150,500)
        self.fc4=nn.Linear(500,331776)

        self.drop1=nn.Dropout(0.3)

    def forward(self,x):
        x=F.tanh(self.fc1(x))
        x=self.drop1(x)
        x=F.tanh(self.fc2(x))
        x=self.drop1(x)
        x=F.tanh(self.fc3(x))
        x=self.drop1(x)
        x=self.fc4(x)

        x=x.view(-1,3,32,72,48)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        

model = mlp()
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
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)#4.69e-4

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
    
    epoch_start=time()
    for k,(x_train,y_train) in enumerate(train_DL):
        optimizer.zero_grad()
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        
        with torch.autocast(device_type="cuda"):
            y_pred = model.forward(x_train)
            batchLoss = lossF(y_pred,y_train)
            
        avgTrainLoss += batchLoss
        
        batchLoss.backward()
        optimizer.step()
        
    train_loss.append(avgTrainLoss/(k+1))
    
    with torch.no_grad():
        avgTestLoss=0
        for k,(x_test,y_test) in enumerate(test_DL):
            x_test=x_test.to(device)
            y_test=y_test.to(device)
            with torch.autocast(device_type="cuda"):
                ypred_val=model.forward(x_test)
                testLoss = lossF(ypred_val, y_test)
                avgTestLoss += testLoss
                
        test_loss.append(avgTestLoss/(k+1))
        
    # Check early stopping condition
    early_stopping.check_early_stop(avgTestLoss)
    
    if early_stopping.stop_training:
        print(f"Early stopping at epoch {i}")
        break

    print(f'Epoch: {i}, train loss: {train_loss[i]}, test loss: {test_loss[i]}, time per epoch: {round(time()-epoch_start,3)}')
          
total_time=time()-start
print(f'Total training time: {total_time/60}')

print('Saving model')
torch.save(model.state_dict(), './results/models/model_MLP_3d.pth')

print('Saving traning curves')
plt.figure(figsize=(10, 6))
plt.plot(torch.Tensor(train_loss), label='Training Loss')
plt.plot(torch.Tensor(test_loss), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('./results/plots/training_losses_MLP_3d.png')
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
np.savez('./results/preds/preds_MLP_3d.npz',ypred_sc)

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
