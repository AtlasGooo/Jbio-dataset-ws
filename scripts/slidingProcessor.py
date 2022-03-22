

from utils import *


class SlidingProcessor:
    
    def __init__(self) -> None:
        pass
    
    def process_relat(self, hip_data:np.ndarray, lf_data:np.ndarray, rf_data:np.ndarray,
                      steps:int, win_step:int, seq_len:int):
        '''
        Process hip_data, lf_data, rf_data to get relative pose between hip-rf-lf
        
        hip_data / lf_data / rf_data : [N,6]
        
        return: 
        
            lf_hip_6D / rf_hip_6D / lf_rf_6D : [N,6]
        
        '''
        
        '''TODO: n't finished or used yet'''
        
        hip = hip_data.copy()
        lf = lf_data.copy()
        rf = rf_data.copy()
        
        lf_hip_6D = np.zeros((steps,))
        
        
        pass
    
    def process_motion(self, xyzrpy_data:np.ndarray, steps:int, win_step:int, seq_len:int) -> np.ndarray:
        '''
        Process [N,6] xyzrpy data to [steps,L,6]. N = frame count
        
        xyzrpy_data: [N,6]
        
        return: [steps,L,6]
        
        Currently, for simplification, only concern about 
        
        1. LF+RF -> Hip. (relative pose in 1s)
        
        Will be modified later.
        
        '''
        
        frame_num = xyzrpy_data.shape[0]
        src_data = xyzrpy_data.copy().reshape((1,frame_num,6))
        trg_data = np.zeros((steps,seq_len,6))  # [steps,L,6]
        

        start = 0
        end = seq_len
        for i in range(steps):
            trg_data[i,:,:] = self.getseq(src_data, start, end) 
            start += win_step
            end += win_step 
            if end >= frame_num:
                break
        # '''(debug)(test)'''        
        # select_step = int(steps*0.5)
        # start = select_step * win_step
        # end = start + seq_len
        # trg_data[select_step,:,:] = self.getseq(src_data, start, end)            
        # print(f'\nstart:{start} end:{end}')
        # print(f'trg_data.shape: {trg_data.shape}')
        # print(f'sequence xyz of step({select_step}): \n{trg_data[select_step,0::10,3:]}')        
        # print(f'src_xyzrpy[start:end,3:]:{src_data[:,start:end:10,3:]}')
            
        
        return trg_data

    def getseq(self, src_data, start, end) -> np.ndarray:
        '''
        get source data and process next sequence length L ([start:end])
        
        src_data: [1,N,6]
        
        return: [1,L,6]
        
        '''
        
        L = end-start
        trg_data = np.zeros((1,L,6))
        
        '''whether it's first sequence'''
        if(start==0):
            last = 0
        else:
            last = start-1
            
        '''get H of last frame'''            
        H_0_last = self.xyzrpy_to_H(src_data[:,last,:])
        R_0_last = H_0_last[:3,:3]
        d_0_last = H_0_last[:3,3].reshape((3,1))
        
        H_last_0 = np.eye(4)
        H_last_0[:3,:3] = R_0_last.T
        H_last_0[:3,3] = np.matmul(-R_0_last.T, d_0_last).reshape((3,))
        
        H_glob = np.zeros((L,4,4))
        H_relat = np.zeros((L,4,4))
        
        
        for i in range(L):
            '''for each frame i in L, get global H_0_i'''
            '''(ATTENTION) remenber the index is start+i instead of i !!!!!!!!'''
            H_0_i = self.xyzrpy_to_H(src_data[:,start+i,:])
            H_glob[i,:,:] = H_0_i
            
            H_last_i = np.matmul(H_last_0,H_0_i)
            H_relat[i,:,:] = H_last_i
            
        
        '''get 6D xyzrpy from relative H'''
        for i in range(L):
            trg_data[:,i,:] = self.H_to_xyzrpy(H_relat[i,:,:])
               
               
        # '''(test)'''
        # check_frame = int(0.5*L)            
        # print(f'\nCheck H_last and H_frame({check_frame}): ')
        # print(f'xyzrpy_last:\n{src_data[:,start-1,:]}')
        # print(f'xyzrpy_frame({check_frame}):\n{src_data[:,start+check_frame,:]})')
        # print(f'H_0_last:\n{H_0_last}')
        # print(f'H_last_0:\n{H_last_0}')
        # print(f'H_glob[check_frame,:,:]:\n{H_glob[check_frame,:,:]}')
        # print(f'H_relat[check_frame,:,:]:\n{H_relat[check_frame,:,:]}')
        # print(f'trg_data[:,check_frame,:]:\n{trg_data[:,check_frame,:]}')
            
            
        return trg_data

        
    def xyzrpy_to_H(self, xyzrpy_data) -> np.ndarray:
        '''
        Get Homogeneous Matrix form 6D vector [x,y,z,rz,ry,rx]
        
        xyzrpy_data: 6D vector (maybe [1,1,6]); euler: 'zyx',degrees=True
        
        return: Homogeneous matrix [4,4]
        
        '''
        src_data = xyzrpy_data.reshape((6,))
        r = R.from_euler('zyx',src_data[-3:],degrees=True)
        d = src_data[:3]      
        Rot = r.as_matrix()
        
        H = np.eye(4)
        H[:3,:3] = Rot
        H[:3,3] = d
        
        return H

    def H_to_xyzrpy(self, H) -> np.ndarray:
        '''
        Get 6D vector [x,y,z,rz,ry,rx] from Homogeneous Matrix
        
        H: 16D vector, maybe [1,4,4]
        
        return: 6D vector [1,1,6]; euler: 'zyx',degrees=True
        '''
        src_H = H.copy().reshape((4,4))
        
        Rot = src_H[:3,:3]
        d = src_H[:3,3].reshape((1,1,3))
        
        r = R.from_matrix(Rot)
        rpy = r.as_euler('zyx',degrees=True).reshape((1,1,3))
        
        trg_data = np.zeros((1,1,6))
        trg_data[:,:,:3] = d
        trg_data[:,:,3:] = rpy
        
        return trg_data
    
    