from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    print('')

def mip(save_dir, npy):
    for i in tqdm(range(npy.shape[2]-4)):
        m = np.nanmax(npy[:,:,i:i+5], 2)
        cv2.imwrite(f'{save_dir}/{i:03}.tif', m)

def slice_3d(save_dir, npy):
    for i in range(npy.shape[2]):
        cv2.imwrite(f'{save_dir}/{i:03}.tif', npy[:,:,i])

def reconstruct(save_path, mips_dir):
    mips_path = Path(mips_dir)
    mips = sorted(mips_path.glob('*.tif'))
    ret = np.zeros((1120,1120,400))
    for idx, mip in enumerate(mips):
        img = cv2.imread(str(mip),0)
        ret[:,:,idx+2]=img

        if idx == 0:
            ret[:,:,0]=img
            ret[:,:,1]=img
        if idx == 395:
            ret[:,:,398]=img
            ret[:,:,399]=img
    
    np.save(save_path, ret)

if __name__ == '__main__':
    iph012 = 'image/2_ON_20170228-165544_IPh006_L_PA797.npy'
    npy = np.load(iph012)
    save = Path('image/iph006_test_mip')
    save.mkdir(exist_ok=True)
    mip(str(save), npy)
    save = Path('image/iph006_test_slice')
    save.mkdir(exist_ok=True)
    slice_3d(str(save), npy)    


    # reconstruct('Result/0809-normal-pseudo/test_pred/pre.npy', Path('Result/0809-normal-pseudo/test_pred/pre'))
    # reconstruct('Result/0809/test_pred/pre.npy', Path('Result/0809/test_pred/pre'))
