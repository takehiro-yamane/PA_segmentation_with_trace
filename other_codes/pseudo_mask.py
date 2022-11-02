from genericpath import exists
import numpy as np
import cv2
from slice_mip import mip, slice_3d
from pathlib import Path

def main(save_path, ):
    print('main')

def bg_pseudo(save_path, pred, th):
    print('bg_pseudo')
    bg = np.where(pred<=th, 255, 0)
    slice_s = save_path/f'bg_pseudo_slice_{th}'
    slice_s.mkdir(exist_ok=True)
    slice_3d(str(slice_s), bg)
    mip_s = save_path/f'bg_pseudo_mip_{th}'
    mip_s.mkdir(exist_ok=True)
    mip(str(mip_s), bg)

def mask(save_path, bg_mip_path, fr_mip_path):
    bgs = sorted(bg_mip_path.glob('*.tif'))
    frs = sorted(fr_mip_path.glob('*.tif'))

    for i, (b, f) in enumerate(zip(bgs, frs)):
        bg = cv2.imread(str(b),0)
        fr = cv2.imread(str(f),0)
        bg = np.where(bg==0, 0, 1)
        fr = np.where(fr==0, 0, 1)
        
        bg = bg.astype('int')
        fr = fr.astype('int')
        tmp = bg | fr
        tmp = np.where(tmp==0, 1, 0)
        tmp*=255
        cv2.imwrite(f'{str(save_path)}/{i:03}.tif', tmp)
    
if __name__ == '__main__':
    # save_path = Path('Result/0808/pseduo')
    # save_path.mkdir(exist_ok=True)
    # pred = np.load('Result/0807/test_pred.npy')
    # bg_pseudo(save_path, pred, 1)

    # bg_mip_path = save_path/f'bg_pseudo_mip_1'
    # fr_mip_path = Path('Result/0808/trace/gaused_s100_60_mip')
    # mask_s = save_path/'mask'
    # mask_s.mkdir(exist_ok=True)
    # mask(mask_s, bg_mip_path, fr_mip_path)



    pred = np.load('Result/0807/test_pred.npy')
    pseudo = np.where(pred>=100, pred, 0)
    save_path = Path('Result/0808/normal-pseudo')
    fr_mip_path = save_path/'pseudo_mip'
    fr_mip_path.mkdir(exist_ok=True)
    mip(str(fr_mip_path),pseudo)
    bg = bg_pseudo(save_path, pred, 1)
    mask_s = Path('Result/0808/normal-pseudo/mask')
    mask_s.mkdir(exist_ok=True)
    bg_mip_path = save_path/'bg_pseudo_mip_1'
    mask(mask_s, bg_mip_path, fr_mip_path)



