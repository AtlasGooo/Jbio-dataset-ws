'''
process raw motion capture data to xyzrpy
the path is in defined in utils.py since they are used in more than one files
'''


from utils import *
from trc import TRCData


SAVE_HIP_PATH = SAVE_DIR + '/hip_xyzrpy.txt'
SAVE_LF_PATH = SAVE_DIR + '/lf_xyzrpy.txt'
SAVE_RF_PATH = SAVE_DIR + '/rf_xyzrpy.txt'


SPEED = 0.5 * 1000
INTV = 1115    # sample interval to print for debug and test


print("Start converting data ...")

'''if the save dir not exist, create the dir'''
if( not os.path.isdir(SAVE_DIR)):
    os.mkdir(SAVE_DIR)

'''read from .trc'''
mocap_data = TRCData()
mocap_data.load(DATA_PATH)

num_frames = int(mocap_data['NumFrames'])
Frames = mocap_data['Frame#']
Time = mocap_data['Time']

marker_hip_back = mocap_data['hip_back']
marker_hip_left = mocap_data['hip_left']
marker_hip_right = mocap_data['hip_right']

marker_lf_back = mocap_data['leftFeet_back']
marker_lf_left = mocap_data['leftFeet_left']
marker_lf_right = mocap_data['leftFeet_right']

marker_rf_back = mocap_data['rightFeet_back']
marker_rf_left = mocap_data['rightFeet_left']
marker_rf_right = mocap_data['rightFeet_right']


'''convert to numpy array'''
np_time = np.array(Time,dtype=np.float32)

np_hip_back = np.array(marker_hip_back)
np_hip_left = np.array(marker_hip_left)
np_hip_right = np.array(marker_hip_right)

np_lf_back = np.array(marker_lf_back)
np_lf_left = np.array(marker_lf_left)
np_lf_right = np.array(marker_lf_right)

np_rf_back = np.array(marker_rf_back)
np_rf_left = np.array(marker_rf_left)
np_rf_right = np.array(marker_rf_right)





'''get middle point'''
np_hip_mid = (np_hip_left + np_hip_right) / 2
np_lf_mid = (np_lf_left + np_lf_right) / 2
np_rf_mid = (np_rf_left + np_rf_right ) / 2



'''add speed in x direction, concatenate back-left-right-mid'''
dx = (np_time * SPEED)
dx = dx.reshape((num_frames,1))

np_hip = np.concatenate((np_hip_back,np_hip_left,np_hip_right,np_hip_mid),axis=1)
np_lf = np.concatenate((np_lf_back,np_lf_left,np_lf_right,np_lf_mid),axis=1)
np_rf = np.concatenate((np_rf_back,np_rf_left,np_rf_right,np_rf_mid),axis=1)

# print(f'check np_hip 1: {np_hip[::1000,:]} \n')

np_hip[:,0::3] += dx
np_lf[:,0::3] += dx
np_rf[:,0::3] += dx

# print(f'check dx: {dx[::INTV,:]}')
# print(f'check np_lf with dx: {np_hip[::INTV,:]} \n')


'''solve vector px and py. px: normalized(B->M), py: normalized(M->L)'''
np_hip_px = np_hip[:,-3:] - np_hip[:,:3]
np_hip_py = np_hip[:,3:6] - np_hip[:,-3:]

np_lf_px = np_lf[:,-3:] - np_lf[:,:3]
np_lf_py = np_lf[:,3:6] - np_lf[:,-3:]

np_rf_px = np_rf[:,-3:] - np_rf[:,:3]
np_rf_py = np_rf[:,3:6] - np_rf[:,-3:]

# print(f'check np_lf_px : {np_lf_px[::INTV,:]} \n')
# print(f'check np_lf_py : {np_lf_py[::INTV,:]} \n')


'''normalized vector'''
np_hip_px = np_hip_px / np.linalg.norm(np_hip_px,axis=1).reshape((num_frames,1))
np_hip_py = np_hip_py / np.linalg.norm(np_hip_py,axis=1).reshape((num_frames,1))

np_lf_px = np_lf_px / np.linalg.norm(np_lf_px,axis=1).reshape((num_frames,1))
np_lf_py = np_lf_py / np.linalg.norm(np_lf_py,axis=1).reshape((num_frames,1))

np_rf_px = np_rf_px / np.linalg.norm(np_rf_px,axis=1).reshape((num_frames,1))
np_rf_py = np_rf_py / np.linalg.norm(np_rf_py,axis=1).reshape((num_frames,1))

# print(f'check np_lf_px normailzed : {np_lf_px[::INTV,:]} \n')
# print(f'check np_lf_py normailzed : {np_lf_py[::INTV,:]} \n')


'''cross product to get z axis, then correct y axis'''
np_hip_pz = np.cross(np_hip_px,np_hip_py)
np_lf_pz = np.cross(np_lf_px,np_lf_py)
np_rf_pz = np.cross(np_rf_px,np_rf_py)

np_hip_py = np.cross(np_hip_pz,np_hip_px)
np_lf_py = np.cross(np_lf_pz,np_lf_px)
np_rf_py = np.cross(np_rf_pz,np_rf_px)

# print(f'check np_lf_pz : {np_lf_pz[::INTV,:]} \n')
# print(f'check corrected np_lf_py : {np_lf_py[::INTV,:]} \n')




'''
solve R_n (global R)
shape: (for the convenience of using scipy.transform)
    the first axis: n (number of frames)
    the second axis: px/py/pz
    the thrid axis: to be concatenate px-py-pz
'''
np_hip_px = np_hip_px.reshape((num_frames,3,1))
np_hip_py = np_hip_py.reshape((num_frames,3,1))
np_hip_pz = np_hip_pz.reshape((num_frames,3,1))

np_lf_px = np_lf_px.reshape((num_frames,3,1))
np_lf_py = np_lf_py.reshape((num_frames,3,1))
np_lf_pz = np_lf_pz.reshape((num_frames,3,1))

np_rf_px = np_rf_px.reshape((num_frames,3,1))
np_rf_py = np_rf_py.reshape((num_frames,3,1))
np_rf_pz = np_rf_pz.reshape((num_frames,3,1))

# print(f'check np_lf_px reshaped : {np_lf_px[::INTV,:]} \n')

R_hip = np.concatenate((np_hip_px,np_hip_py,np_hip_pz),axis=2)
R_lf = np.concatenate((np_lf_px,np_lf_py,np_lf_pz),axis=2)
R_rf = np.concatenate((np_rf_px,np_rf_py,np_rf_pz),axis=2)


# print(f'check R_lf concated : \n{R_lf[::INTV,:,:]} \n')


'''convert to scipy transform, say, to euler'''
scipy_R_hip = R.from_matrix(R_hip)
scipy_R_lf = R.from_matrix(R_lf)
scipy_R_rf = R.from_matrix(R_rf) 

eul_hip = scipy_R_hip.as_euler('zyx',degrees=True)
eul_lf = scipy_R_lf.as_euler('zyx',degrees=True)
eul_rf = scipy_R_rf.as_euler('zyx',degrees=True)


print(f'eul_lf sample : \n{eul_lf[::INTV,:]}\n')


'''
concate xyz(back) and rpy and write file
shape: [n,6]
note that the unit of xyz is mm, rpy is angle
'''

hip_xyzrpy = np.concatenate((np_hip[:,:3],eul_hip),axis=1)
lf_xyzrpy = np.concatenate((np_lf[:,:3],eul_lf),axis=1)
rf_xyzrpy = np.concatenate((np_rf[:,:3],eul_rf),axis=1)

print(f'final concat lf_xyzrpy : \n{lf_xyzrpy[::INTV,:]}\n')

np.savetxt(SAVE_HIP_PATH,hip_xyzrpy,fmt='%.10e',delimiter=',')
np.savetxt(SAVE_LF_PATH,lf_xyzrpy,fmt='%.10e',delimiter=',')
np.savetxt(SAVE_RF_PATH,rf_xyzrpy,fmt='%.10e',delimiter=',')

print("The end. Success.")



