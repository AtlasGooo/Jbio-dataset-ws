
import imp
from utils import *
from sklearn.model_selection import train_test_split

class DataImporter:

    def __init__(self) -> None:
        pass

    def prepare_data_1(self,hip_seq_1,lf_seq_1,rf_seq_1,
                    hip_seq_2,lf_seq_2,rf_seq_2,
                    hip_seq_3,lf_seq_3,rf_seq_3):

        '''
        uniformly split 0.5,0.8,1.1 to train and val
        
        only motion of hip/lf/rf itself were estimated
        
        '''

        hip_seq = np.concatenate((hip_seq_1,hip_seq_2,hip_seq_3),axis=0)
        lf_seq = np.concatenate((lf_seq_1,lf_seq_2,lf_seq_3),axis=0)
        rf_seq = np.concatenate((rf_seq_1,rf_seq_2,rf_seq_3),axis=0) 

        '''try to scale the data with unit (n't use normalization here)'''
        hip_seq[:,:3] *= 0.001
        lf_seq[:,:,3] *= 0.001
        rf_seq[:,:,3] *= 0.001
        
        hip_seq[:,3:] *= (np.pi/180)
        lf_seq[:,3:] *= (np.pi/180)    
        rf_seq[:,3:] *= (np.pi/180)
        
        
        '''concat lf_seq and rf_seq, data size 6->12'''
        np_inputs = np.concatenate((lf_seq,rf_seq),axis=2)  #[N,L,12]
        np_labels = hip_seq     #[N,6]
        
        
        '''split into train and test/validate set'''
        train_x, val_x, train_y, val_y = train_test_split(
            np_inputs, np_labels, test_size=0.1, random_state=66, shuffle=True)
        
        return train_x, train_y, val_x, val_y

    def prepare_data_2(self,hip_seq_1,lf_seq_1,rf_seq_1,
                    hip_seq_2,lf_seq_2,rf_seq_2,
                    hip_seq_3,lf_seq_3,rf_seq_3):
        '''
        use 0.5 and 1.1 to train , predict 0.8
        
        seq1: 0.5, seq2: 0.8, seq3: 1.1
        
        only motion of hip/rf/lf itself were estimated
        
        '''    

        hip_seq = np.concatenate((hip_seq_1,hip_seq_3),axis=0)  # data: 0.5,1.1 label:0.8
        lf_seq = np.concatenate((lf_seq_1,lf_seq_3),axis=0)
        rf_seq = np.concatenate((rf_seq_1,rf_seq_3),axis=0) 

        hip_seq_val = hip_seq_2.copy()
        lf_seq_val = lf_seq_2.copy()
        rf_seq_val = rf_seq_2.copy()

        '''try to scale the data with unit (n't use normalization here)'''
        hip_seq[:,:3] *= 0.001
        lf_seq[:,:,3] *= 0.001
        rf_seq[:,:,3] *= 0.001
        hip_seq[:,3:] *= (np.pi/180)
        lf_seq[:,3:] *= (np.pi/180)    
        rf_seq[:,3:] *= (np.pi/180)
        
        hip_seq_val[:,:3] *= 0.001
        lf_seq_val[:,:,3] *= 0.001
        rf_seq_val[:,:,3] *= 0.001
        hip_seq_val[:,3:] *= (np.pi/180)
        lf_seq_val[:,3:] *= (np.pi/180)    
        rf_seq_val[:,3:] *= (np.pi/180)
        
        
        '''concat lf_seq and rf_seq, data size 6->12'''
        np_inputs = np.concatenate((lf_seq,rf_seq),axis=2)  #[N,L,12]
        np_labels = hip_seq     #[N,6]
        
        np_inputs_val = np.concatenate((lf_seq_val,rf_seq_val),axis=2)  #[N,L,12]
        np_labels_val = hip_seq_val     #[N,6]
        
        '''split into train and test/validate set'''
        train_x, _, train_y, _ = train_test_split(
            np_inputs, np_labels, test_size=0.01, random_state=66, shuffle=True)
        
        _, val_x, _, val_y = train_test_split(
            np_inputs_val, np_labels_val, test_size=0.99, random_state=66, shuffle=True)
        
        
        return train_x, train_y, val_x, val_y        

        
