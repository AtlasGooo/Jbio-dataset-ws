
from sklearn.model_selection import train_test_split
from utils import *
from GRUNet import *
from dataImporter import *

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader




if __name__ == '__main__':
    
    DIR_1 = ROOT_DIR + '/processed_0.5_1'
    DIR_2 = ROOT_DIR + '/processed_0.8_11'
    DIR_3 = ROOT_DIR + '/processed_1.1_11_3300_12100'
    DIR_4 = ROOT_DIR + '/processed_0.8_21'
    
    HIP_1 = DIR_1 + '/hip_seq.npy'
    LF_1 = DIR_1 + '/lf_seq.npy'
    RF_1 = DIR_1 + '/rf_seq.npy' 
    
    HIP_2 = DIR_2 + '/hip_seq.npy'
    LF_2 = DIR_2 + '/lf_seq.npy'
    RF_2 = DIR_2 + '/rf_seq.npy' 
    
    HIP_3 = DIR_3 + '/hip_seq.npy'
    LF_3 = DIR_3 + '/lf_seq.npy'
    RF_3 = DIR_3 + '/rf_seq.npy'
    
    HIP_4 = DIR_4 + '/hip_seq.npy'
    LF_4 = DIR_4 + '/lf_seq.npy'
    RF_4 = DIR_4 + '/rf_seq.npy'
    
    BATCH_SIZE = 128    
    ADAM_LR = 0.005  # init val for dynamic learning rate. see adjust_lr() in GRUNet.py.

    EPOCHS = 30

    
    '''load data, data size: (N,L,Hin)'''
    print(f'Loading data ......\n')
    hip_seq_1 = np.load(HIP_1)     #[N,6]
    lf_seq_1 = np.load(LF_1)       #[N,L,6]
    rf_seq_1 = np.load(RF_1)       #[N,L,6]
    
    hip_seq_2 = np.load(HIP_2)     #[N,6]
    lf_seq_2 = np.load(LF_2)       #[N,L,6]
    rf_seq_2 = np.load(RF_2)       #[N,L,6]
    
    hip_seq_3 = np.load(HIP_3)     #[N,6]
    lf_seq_3 = np.load(LF_3)       #[N,L,6]
    rf_seq_3 = np.load(RF_3)       #[N,L,6]
         
    data_importer = DataImporter()
    train_x, train_y, val_x, val_y = data_importer.prepare_data_2(
        hip_seq_1,lf_seq_1,rf_seq_1,
        hip_seq_2,lf_seq_2,rf_seq_2,
        hip_seq_3,lf_seq_3,rf_seq_3)  
    

    '''(test) for simple case, only a few data is enough'''
    # if(train_x.shape[0] >= 641):
    #     train_x = train_x[:641,:,:]
    #     train_y = train_y[:641,:]


    # '''(test)'''
    print(f'train_x.shape:{train_x.shape}, train_y.shape:{train_y.shape}')
    print(f'val_x.shape:{val_x.shape}, val_y.shape:{val_y.shape}')
    
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    print(f'Loading data finished.\n')   



    '''training loop'''
    print(f'Start trainning ......\n')
    input_dim = train_x.shape[2]
    output_dim = train_y.shape[1]
    
    
    model, train_loss_arr, val_loss_arr, epoch_times = \
        train_and_val(
        train_loader=train_loader, 
        val_loader=test_loader, 
        train_batch_size=BATCH_SIZE, 
        val_batch_size=BATCH_SIZE, 
        input_dim=input_dim, 
        output_dim=output_dim, 
        learn_rate=ADAM_LR, 
        hidden_dim=256, 
        EPOCHS=EPOCHS)
    
    
    print(f'Trainning finished.\n')    
    
    
    '''plot training process'''
    fig,axes = plt.subplots(2,1)

    plot_x = np.arange(EPOCHS)
    plot_y1 = np.array(train_loss_arr)
    plot_y2 = np.array(val_loss_arr)
    
    last = int(0.3*EPOCHS)
    intv = max(int(0.01*EPOCHS),1)
    
    axes[0].plot(plot_x[::intv],plot_y1[::intv],'bo-',plot_x[::intv],plot_y2[::intv],'ro-')    
    axes[1].plot(plot_x[-last::intv],plot_y1[-last::intv],'bo-',plot_x[-last::intv],plot_y2[-last::intv],'ro-')
    
    # plt.plot(plot_x[-last:],plot_y1[-last:],'bo--',plot_x[-last:],plot_y2[-last:],'ro--')
    
    
    # plt.pause(0)
    plt.waitforbuttonpress()
    





