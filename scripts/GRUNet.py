'''
GRU input : in GRU, batch_first=True, (N,L,Hin), L sequence length, Hin data size
GRU output : in GRU, batch_first=True, (N,L,D * Hout), D means bidirectional
'''


from utils import *

import torch
from torch import nn
import time





# DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.fc0 = nn.Linear(input_dim, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):    
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(DEVICE)
        return hidden




def adjust_lr(optimizer, total_epoch, current_epoch, init_lr, init_loss):
    lr = init_lr * ( 0.5 ** (current_epoch // (0.1*total_epoch)) )
    lr = max(lr,init_lr*0.005)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_and_val(
    train_loader, 
    val_loader, 
    train_batch_size, 
    val_batch_size, 
    input_dim, 
    output_dim, 
    learn_rate, 
    hidden_dim=256, 
    EPOCHS=5):


    # input_dim = next(iter(train_loader))[0].shape[2]    # may raise error due to batch size and data len
    
    '''(test)'''
    print(f'\nTrain loader len:{len(train_loader)}, val loader len:{len(val_loader)}')
    
    n_layers = 2

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2)
    model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    model.train()
    epoch_times = []
    train_loss_arr = []
    val_loss_arr = []
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        adjust_lr(optimizer, EPOCHS, epoch, learn_rate, 0)
        
        train_loss = train(model, train_loader, train_batch_size, optimizer, criterion)
        val_loss = validate(model, val_loader, val_batch_size, criterion)
          
        train_loss_arr.append(train_loss/len(train_loader))
        val_loss_arr.append(val_loss/len(val_loader))
        
        current_time = time.time()
        if ( epoch <= 5):
            print(f"\nEpoch {epoch}/{EPOCHS} done.")
            print(f'Avg train loss: {train_loss/len(train_loader)}, Avg val loss: {val_loss/len(val_loader)}')
            print(f"Time Elapsed for Epoch: {current_time-start_time} seconds")
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print(f'Modified learning rate: {lr}')            
        elif( EPOCHS >= 20 and epoch % (EPOCHS/10)==0 ):
            print(f"\nEpoch {epoch}/{EPOCHS} done.")
            print(f'Avg train loss: {train_loss/len(train_loader)}, Avg val loss: {val_loss/len(val_loader)}')
            print(f"Time Elapsed for Epoch: {current_time-start_time} seconds")
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                print(f'Modified learning rate: {lr}')
        else:
            pass
        
        epoch_times.append(current_time-start_time)
        
    print(f"\nTotal Training Time: {sum(epoch_times)} seconds")
    
    
    return model, train_loss_arr, val_loss_arr, epoch_times      
        
        
        
def train(model, train_loader, train_batch_size, optimizer, criterion):
    model.train()
    h = model.init_hidden(train_batch_size)
    epoch_loss = 0.   
     
    for x, label in train_loader:
        h = h.data
        model.zero_grad()
        out, h = model(x.to(DEVICE).float(), h)         
        loss = criterion(out, label.to(DEVICE).float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()    

    return epoch_loss

def validate(model, val_loader, val_batch_size, criterion):
    model.eval()
    h = model.init_hidden(val_batch_size)
    epoch_loss = 0.
    with torch.no_grad():
        for x, label in val_loader:
            h = h.data
            out, h = model(x.to(DEVICE).float(), h)         
            loss = criterion(out, label.to(DEVICE).float())

            epoch_loss += loss.item()
    
    return epoch_loss

    
def evaluate():
    pass
