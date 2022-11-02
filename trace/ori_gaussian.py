from random import gauss
# import cc3d
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm
from pathlib import Path
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import copy
from scipy.ndimage.filters import gaussian_filter
import math

from other_codes import slice_mip

def main(save_dir,skel_npy, trace_th, seed_th):
    print('main')
    c_list = point_list(skel_npy)
    gaus_npy = gaussian_pal(skel_npy, c_list)
    gaus_npy = gaus_npy*255
    np.save(f'{str(save_dir)}/gaused_s{seed_th}_{trace_th}',gaus_npy)

    slice_save = save_dir/f'gaused_s{seed_th}_{trace_th}'
    slice_save.mkdir(exist_ok=True)
    for i in range(gaus_npy.shape[2]):
        cv2.imwrite(f'{str(slice_save)}/{i:03}.tif', gaus_npy[:,:,i])

    mip_save = save_dir/f'gaused_s{seed_th}_{trace_th}_mip'
    mip_save.mkdir(exist_ok=True)
    slice_mip.mip(str(mip_save), gaus_npy)


#前景の座標のリストを返す
def point_list(npy):
    coor = np.where(npy!=0)
    c_list = []
    for i in range(len(coor[0])):
        tmp_list = []
        tmp_list.append(coor[0][i])
        tmp_list.append(coor[1][i])
        tmp_list.append(coor[2][i])
        c_list.append(tmp_list)
    return c_list

def gaussian_pal(npy, c_list):
    n = len(c_list)
    q = int(n/4)+1

    s = npy.shape
    
    with ThreadPoolExecutor(10) as e:
        r0 = e.submit(gaussian, s, c_list[:q])
        r1 = e.submit(gaussian, s, c_list[q:q*2])
        r2 = e.submit(gaussian, s, c_list[q*2:q*3])
        r3 = e.submit(gaussian, s, c_list[q*3:])
    rets = [r0, r1, r2, r3]
    ret = np.zeros_like(npy)
    for i in rets:
        ret = np.maximum(ret, i.result())
    return ret


def gaussian(s,c_list):
    test = np.zeros((5,5,5))
    test[2][2][2] = 1
    a = gaussian_filter(test, sigma=1)
    kernel = a/a.max()
    ret = np.zeros(s)
    for c in tqdm(c_list,leave=False):
        # tmp = np.zeros_like(ret)
        tmp_kernel = kernel
        x = c[0]-2
        if x<0:
            tmp_kernel = tmp_kernel[abs(x):, :, :]
            x=0
        if x+5>ret.shape[0]:
            m = x+5-ret.shape[0]
            tmp_kernel = tmp_kernel[:5-m,:,:]
        y = c[1]-2
        if y<0:
            tmp_kernel = tmp_kernel[:, abs(y):, :]
            y=0
        if y+5>ret.shape[1]:
            m = y+5-ret.shape[1]
            tmp_kernel = tmp_kernel[:,:5-m,:]
        z = c[2]-2
        if z<0:
            tmp_kernel = tmp_kernel[:,:,abs(z):]
            z=0
        if z+5>ret.shape[2]:
            m = z+5-ret.shape[2]
            tmp_kernel = tmp_kernel[:,:,:5-m]

        # tmp[x:x+tmp_kernel.shape[0],y:y+tmp_kernel.shape[1],z:z+tmp_kernel.shape[2]]=tmp_kernel

        ret_tmp = ret[x:x+tmp_kernel.shape[0],y:y+tmp_kernel.shape[1],z:z+tmp_kernel.shape[2]]
        ret_tmp = np.maximum(ret_tmp,tmp_kernel)
        ret[x:x+tmp_kernel.shape[0],y:y+tmp_kernel.shape[1],z:z+tmp_kernel.shape[2]] = ret_tmp
    return ret
    

if __name__ == '__main__':
    skel_npy = 'trace/traced_skel_s120_60.npy'
    skel_npy = np.load(skel_npy)
    save_dir = Path('')
    main(save_dir, skel_npy, trace_th=60, seed_th=120)

