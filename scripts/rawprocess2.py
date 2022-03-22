'''
process xyzrpy to sliding window batch data
for both mocap and imu data

data before processing (from rawprocess1.py):
(N,6)

processed data:
lf/rf(input): [N,L,6] (or [steps,L,6]), see GRUNet's input
hip(output): [N,6]

'''


from utils import *
from slidingProcessor import SlidingProcessor

HIP_DATA = SAVE_DIR + '/hip_xyzrpy.txt'
LF_DATA = SAVE_DIR + '/lf_xyzrpy.txt'
RF_DATA = SAVE_DIR + '/rf_xyzrpy.txt'

SAVE_HIP_SEQ_DATA = SAVE_DIR + '/hip_seq.npy'
SAVE_LF_SEQ_DATA = SAVE_DIR + '/lf_seq.npy'
SAVE_RF_SEQ_DATA = SAVE_DIR + '/rf_seq.npy'


MOCAP_FRAME_RATE = 120
WIN_LEN = int(MOCAP_FRAME_RATE*2)     # length for 1s
WIN_STEP = int(MOCAP_FRAME_RATE * 0.1)      



hip_data = np.loadtxt(HIP_DATA,dtype=float,comments='#',delimiter=',')
lf_data = np.loadtxt(LF_DATA,dtype=float,comments='#',delimiter=',')
rf_data = np.loadtxt(RF_DATA,dtype=float,comments='#',delimiter=',')

frame_num = hip_data.shape[0]




'''
Do sliding window on data [N,6], generate new data [steps,L,6]

Currently, for simplification, only concern about 

1. LF+RF -> Hip. (relative pose in 1s)

Will be modified later.

'''
sliding = SlidingProcessor()

steps = int( (frame_num-WIN_LEN) / WIN_STEP )


# '''(debug)'''
# print(f'steps: {steps}')
# if(steps > 1000):
#     steps = 1000
# print(f'reduced steps for testing: {steps}')


'''(test)'''
print(f'\ntotal steps: {steps}')
print(f'\nprocessing hip data ...')
hip_seq_data = sliding.process_motion(hip_data, steps, WIN_STEP, WIN_LEN)  # [steps,L,6]
'''(test)'''
print(f'\nprocessing lf data ...')
lf_seq_data = sliding.process_motion(lf_data, steps, WIN_STEP, WIN_LEN)
'''(test)'''
print(f'\nprocessing rf data ...')
rf_seq_data = sliding.process_motion(rf_data, steps, WIN_STEP, WIN_LEN)


'''for hip, only need the last vector (last pose) in each sequence'''
hip_data = hip_seq_data[:,-1,:]


np.save(SAVE_HIP_SEQ_DATA,hip_data)     # [steps,6]
np.save(SAVE_LF_SEQ_DATA,lf_seq_data)   # [steps,L,6]
np.save(SAVE_RF_SEQ_DATA,rf_seq_data)   # [steps,L,6]



'''(test)'''
print(f'test data:')
print(f'hip/lf/rf:  {hip_data.shape} / {lf_seq_data.shape}')

for i in range(5): 
    step_i = i*10

    print(f'\ncheck xyz of step {step_i}:')
    print(f'hip: \n{hip_data[step_i,:3]}')
    print(f'lf: \n{lf_seq_data[step_i,0::30,:3]}')
    print(f'rf: \n{rf_seq_data[step_i,0::30,:3]}')
    
    print(f'\ncheck rpy("zyx") of step {step_i}:')
    print(f'hip: \n{hip_data[step_i,3:]}')
    print(f'lf: \n{lf_seq_data[step_i,0::30,3:]}')
    print(f'rf: \n{rf_seq_data[step_i,0::30,3:]}')   
    
    print("\n\n")


print("The end. Success.")

