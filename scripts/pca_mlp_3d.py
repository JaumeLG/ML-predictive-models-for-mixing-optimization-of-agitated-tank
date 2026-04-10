import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split,Subset

import numpy as np
import matplotlib.pyplot as plt
from math import exp

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from time import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
valSplit=0.8
epochs=5000
lr=1e-4
batch=32

nM=420

# Basic setup for early stopping criteria
patience = 500  # epochs to wait after no improvement
delta = 0.005  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 500  # count of epochs with no improvement


print('Loading data')
inputsOrig=np.load('./dataset/inputsPCA.npz')['arr_0']
outputsOrig=np.load('./dataset/outputsPCA.npz')['arr_0']

print(inputsOrig.shape,outputsOrig.shape)


print('Performing PCA to reduce dimensionality')
mean=np.mean(outputsOrig,axis=0)
outputsOrig=outputsOrig - mean

#perform PCA
pca = PCA(n_components=nM)   
outputsPCA = pca.fit_transform(outputsOrig)
print(outputsPCA.shape)

print('Scaling data')
scalerIn=StandardScaler()
inputs_sc=scalerIn.fit_transform(inputsOrig)

outputs_sc=outputsPCA

print('Creating dataset')
train_size = int(inputs_sc.shape[0]*valSplit)
inputs_sc=torch.Tensor(inputs_sc)
outputs_sc=torch.Tensor(outputs_sc)

dataset=TensorDataset(inputs_sc,outputs_sc)

val_size = inputs_sc.shape[0] - train_size -10
usableIndexes = list(range(len(dataset)-10))
usableSet = Subset(dataset,usableIndexes)
trainData, testData = random_split(usableSet,[train_size, val_size],generator=torch.Generator().manual_seed(37))

train_DL = DataLoader(trainData,batch_size=batch,pin_memory=True,shuffle=True)
test_DL = DataLoader(testData,batch_size=8,pin_memory=True,shuffle=False)


device=torch.device('cuda')

class mlp(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(inputs_sc.shape[1], 32)
        self.fc2 = nn.Linear(32,128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, nM)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):


        x = F.elu(self.fc1(x),alpha=0.2)
        x = self.drop(x)
        x = F.elu(self.fc2(x),alpha=0.2)
        x = self.drop(x)
        x = F.elu(self.fc3(x),alpha=0.2)
        x = self.drop(x)
        #x = F.elu(self.fc4(x),alpha=0.2)
        x = self.fc4(x)

        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        
model = mlp()
model = model.to(device)
model.apply(init_weights)

print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.std=torch.Tensor(pca.explained_variance_)
        self.weight= 1 / self.std.to(device)
        
    def forward(self, pred, target):
        loss = ((target - pred)**2 * self.weight).mean()
        return loss

lossF = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)#4.69e-4

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
            
        avgTrainLoss += batchLoss.item()
        
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
                avgTestLoss += testLoss.item()
                
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
torch.save(model.state_dict(), './results/models/model_pca_mlp_3d.pth')

print('Saving traning curves')
plt.figure(figsize=(10, 6))
plt.plot(torch.Tensor(train_loss), label='Training Loss')
plt.plot(torch.Tensor(test_loss), label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig('./results/plots/training_losses_pca_mlp_3d.png')
plt.close()
#plt.show()

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


print('Scaling predictions')
testOrig=pca.inverse_transform(ytest)
testOrig=testOrig + mean

print(ytest.shape,testOrig.shape)

predOrig=pca.inverse_transform(ypred_val)
predOrig=predOrig + mean

error0=np.mean(abs(predOrig-testOrig))
error1=np.mean(abs(predOrig-testOrig)/(testOrig+1e-5))
error2=np.mean(abs(predOrig-testOrig)/np.std(testOrig,axis=0))
error3=np.mean(abs(predOrig-testOrig)/np.max(testOrig,axis=0))

print(f'MAE: {error0}, RelErr: {error1}, RelErr_std: {error2}, RelErr_inf: {error3}')

#To visualize some predictions
cases=testOrig.shape[0]
velTest=testOrig.reshape(cases,-1,3)
velTest=np.transpose(velTest,(0,2,1))
velTest=velTest.reshape(cases,3,32,-1).reshape(cases,3,32,72,48)
velPred=predOrig.reshape(cases,-1,3)
velPred=np.transpose(velPred,(0,2,1))
velPred=velPred.reshape(cases,3,32,-1).reshape(cases,3,32,72,48)

for i in range(10):
    plt.subplot(1,2,1)
    plt.imshow(velTest[i,0,15,:,:])
    plt.xticks([])
    plt.yticks([])
    plt.title('Ground truth')

    plt.subplot(1,2,2)
    plt.imshow(velPred[i,0,15,:,:])
    plt.xticks([])
    plt.yticks([])
    plt.title('Predicted')

    plt.show()

print('Shape preds:',velPred.shape)
np.savez('./results/preds/preds_pca_mlp_3d.npz',velPred)

pred_mod=np.sqrt(np.sum(velPred**2,axis=1))
true_mod=np.sqrt(np.sum(velTest**2,axis=1))

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
