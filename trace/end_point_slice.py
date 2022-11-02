import cc3d
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import copy
from scipy.ndimage.filters import gaussian_filter
import math
'''
parammeters
binarize_threshold
'''

def main(save_dir,npy_path, th):
    print('main')
    ori = np.load(npy_path)
    npy = np.where(ori>=th, 255, 0)
    labels = cc3d.connected_components(npy)
    # print(labels.max())

    #test
    # ep_list = end_points(10, 20, npy, labels)
    # ep_npy = plot_endpoints(0, len(ep_list), ep_list, ori)

    ep_list = end_points_pal(npy, labels)
    df = pd.DataFrame(ep_list)
    df.to_csv(f'{str(save_dir)}/ep_list.csv')

    # df = pd.read_csv(f'{str(save_dir)}/ep_list.csv')
    # df = df.drop(df.columns[[0]], axis=1)
    # ep_list = df.values.tolist()

    ep_npy = plot_endpoints_pal(ep_list, ori)
    np.save(f'{str(save_dir)}/endpoints', ep_npy)

    save = save_dir/'endpoint_slice'
    save.mkdir(exist_ok = True)
    for i in range(ep_npy.shape[2]):
        cv2.imwrite(f'{str(save)}/{i:03}.tif', ep_npy[:,:,i])


    denoised, skel = denoise_skel_image(npy, labels)
    np.save(f'{str(save_dir)}/skeled.npy',skel)
    save_path_d = save_dir/'denoised_slice'
    save_path_s = save_dir/'skeled_slice'
    save_path_d.mkdir(exist_ok=True)
    save_path_s.mkdir(exist_ok=True)
    for i in range(denoised.shape[2]):
        cv2.imwrite(f'{str(save_path_d)}/{i:03}.tif', denoised[:,:,i])
    for i in range(skel.shape[2]):
        cv2.imwrite(f'{str(save_path_s)}/{i:03}.tif', skel[:,:,i])


def end_points_slice(save_path, ep_npy_path, npy_for_slice,th):
    print('end point slice')


    ori = np.load(ep_npy_path)
    npy = np.where(ori>=th, 255, 0)
    labels = cc3d.connected_components(npy)

    npy_s = np.load(npy_for_slice)

    ep_list = end_points_pal(npy, labels)
    df = pd.DataFrame(ep_list)
    df.to_csv(f'{str(save_path)}/ep_list.csv')


    for id, i in enumerate(tqdm(ep_list)):
        ep = [i[0],i[1],i[2]]
        vec = [i[3], i[4], i[5]]
        label_id = i[6]
        tmp = ori[labels==label_id]
        max = tmp.max()
        min = tmp.min()
        ave = tmp.mean()
        intensity = [max, min, ave]
        slice_ep(save_path, ep, vec, npy_s, id, intensity)

def info(save_path, ep, vec, int):
    slice_vec = 'yz'
    if not 0 in vec:
        if vec[1]==-1:
            slice_vec = 'zx'
        elif vec[2]==-1:
            slice_vec = 'xy'
    else:
        if vec[0]!=0:
            slice_vec = 'yz'
        elif vec[1]!=0:
            slice_vec = 'zx'
        elif vec[2]!=0:
            slice_vec = 'xy'
        

    with open(f'{str(save_path)}/info.txt', 'w') as f:
        f.write(f'coordinates : {ep[0]},{ep[1]},{ep[2]}\n')
        f.write(f'vectors : {vec[0]},{vec[1]},{vec[2]}\n')
        f.write(f'best slice direction : {slice_vec}\n')
        f.write(f'intensity max : {int[0]}\n')
        f.write(f'intensity min : {int[1]}\n')
        f.write(f'intensity ave : {int[2]}\n')

def slice_ep(save_path, ep, vec, ori, id, intensity):
    #xyz=0 mean slice yz
    #xyz=1 mean slice zx
    #xyz=2 mean slice xy
    save_path = save_path/f'{id:03}'
    save_path.mkdir(exist_ok = True)

    info(save_path, ep, vec, intensity)

    sp0 = save_path/'yz'
    sp1 = save_path/'zx'
    sp2 = save_path/'xy'
    sp0.mkdir(exist_ok=True)
    sp1.mkdir(exist_ok=True)
    sp2.mkdir(exist_ok=True)

    x_min = ep[0]-50
    x_max = ep[0]+50
    y_min = ep[1]-50
    y_max = ep[1]+50
    z_min = ep[2]-50
    z_max = ep[2]+50

    if x_min<0:
        x_min = 0
    if y_min<0:
        y_min = 0
    if z_min<0:
        z_min = 0
    if x_max > ori.shape[0]:
        x_max = ori.shape[0]
    if y_max > ori.shape[1]:
        y_max = ori.shape[1]
    if z_max > ori.shape[2]:
        z_max = ori.shape[2]

    v = 1
    for xyz in range(3):
        if vec[xyz]==0:
            v = 1
        else:
            v = vec[xyz]
        t = ep[xyz] #slice point
        s = t-v
        e = t+14*v
        for d, i in enumerate(range(s, e, v)):
            if xyz==0:
                if i>=ori.shape[0] or i<0:
                    continue
                cv2.imwrite(f'{str(sp0)}/{d:03}.tif', ori[i,y_min:y_max,z_min:z_max])
            elif xyz==1:
                if i>=ori.shape[1] or i<0:
                    continue
                cv2.imwrite(f'{str(sp1)}/{d:03}.tif', ori[x_min:x_max,i,z_min:z_max])
            elif xyz==2:
                if i>=ori.shape[2] or i<0:
                    continue
                cv2.imwrite(f'{str(sp2)}/{d:03}.tif', ori[x_min:x_max,y_min:y_max,i])


def plot_ep(ep_list, ori):
    ret = np.zeros_like(ori)

    for i in range(1, len(ep_list)):
        kernel = np.ones((3,3,3))
        tmp = np.zeros_like(ori)
        x = ep_list[i][0]
        y = ep_list[i][1]
        z = ep_list[i][2]
        


def plot_endpoints_pal(ep_list, ori):
    ret = np.zeros_like(ori)
    list_num = len(ep_list)
    q = int(list_num/3)+1

    o_s = ori.shape
    with ThreadPoolExecutor(20) as e:
        r01 = e.submit(plot_endpoints, 1, q, ep_list, o_s)
        r02 = e.submit(plot_endpoints, q, q*2, ep_list, o_s)
        r03 = e.submit(plot_endpoints, q*2, list_num, ep_list, o_s)

    ret = np.maximum(r01.result(), r02.result())
    ret = np.maximum(ret, r03.result())
    return ret

def plot_endpoints(s, g, ep_list, ori_shape):
    print('plot end points')
    ret = np.zeros((ori_shape))

    # kernel = original_gaussian_kernel()
    kernel = np.ones((3,3,3))

    for i in tqdm(range(s, g)):

        npy = np.zeros((ori_shape))
        ep = ep_list[i]
        x0 = ep[0]-1
        x1 = ep[0]+2
        y0 = ep[1]-1
        y1 = ep[1]+2
        z0 = ep[2]-1
        z1 = ep[2]+2

        tmp_kernel = np.copy(kernel)
        if ep[0]==0:
            tmp_kernel = np.delete(tmp_kernel, 0, 0)
            x0 += 1
        if ep[1]==0:
            tmp_kernel = np.delete(tmp_kernel, 0, 1)
            y0 += 1
        if ep[2]==0:
            tmp_kernel = np.delete(tmp_kernel, 0, 2)
            z0 += 1
        if ep[0]==npy.shape[0]-1:
            tmp_kernel = np.delete(tmp_kernel, 2, 0)
            x1 -= 1
        if ep[1]==npy.shape[1]-1:
            tmp_kernel = np.delete(tmp_kernel, 2, 1)
            y1 -= 1
        if ep[2]==npy.shape[2]-1:
            tmp_kernel = np.delete(tmp_kernel, 2, 2)
            z1 -= 1

        # npy[x0:x1, y0:y1, z0:z1] = tmp_kernel*255

        # ret = np.maximum(ret, npy)

        ret[x0:x1, y0:y1, z0:z1] = tmp_kernel*255

        if math.isnan(ret[0,0,0]):
            raise ValueError("not a number")

    return ret

def original_gaussian_kernel():
    kernel = np.ones((3,3,3))
    
    d2 = [[1,0,1],[1,1,0],[1,1,2],[1,2,1],[0,1,1],[2,1,1]]
    d4 = [[1,0,0],[1,0,2],[1,2,0],[1,2,2],[0,0,1],[0,2,1],[2,0,1],[2,2,1],[0,1,0],[0,1,2],[2,1,0],[2,1,2]]
    d8 = [[0,0,0],[0,0,2],[0,2,0],[0,2,2],[2,0,0],[2,0,2],[2,2,0],[2,2,2]]

    for i in d2:
        kernel[i[0],i[1],i[2]]/=2
    for i in d4:
        kernel[i[0],i[1],i[2]]/=4
    for i in d8:
        kernel[i[0],i[1],i[2]]/=8
    
    return kernel

def ret_extend(rets):
    ret_list = []
    for ret in rets:
        ret_list.extend(ret.result())
    return ret_list

def end_points_pal(npy, labels):
    label_num = labels.max()+1
    q = int (label_num/5)+1

    with ThreadPoolExecutor(20) as e:
        r01 = e.submit(end_points, 1, q, npy, labels)
        r02 = e.submit(end_points, q, q*2, npy, labels)
        r03 = e.submit(end_points, q*2, q*3, npy, labels)
        r04 = e.submit(end_points, q*3, q*4, npy, labels)
        r05 = e.submit(end_points, q*4, label_num, npy, labels)

    rets = [r01 ,r02 ,r03, r04, r05]
    ret_list = ret_extend(rets)
    ret_list = get_unique_list(ret_list)
    return ret_list

def end_points(s, g, npy, labels):
    ep_list = []
    for i in tqdm(range(s, g), leave=False):
        tmp = npy[labels==i]
        if len(tmp)<30:
            continue
        
        tmp_npy = np.where(labels==i, 1, 0)
        a = np.where(labels==i)
        x_min = a[0].min()
        x_max = a[0].max()
        y_min = a[1].min()
        y_max = a[1].max()
        z_min = a[2].min()
        z_max = a[2].max()
        
        tmp_npy_1=tmp_npy[x_min:x_max, y_min:y_max, z_min:z_max]
        pad_npy = np.pad(tmp_npy_1, ((1,1)))
        ##細線化
        ske_npy = skeletonize(pad_npy)

        ep_list.extend(judge_end(ske_npy, x_min, y_min, z_min, i))
    return ep_list

def judge_end(npy, x, y, z, label_id):
    j_list = judge_list()
    vt0, vt1, vt2 = vec_table()
    endlist = []
    for i in range(npy.shape[0]-2):
        for j in range(npy.shape[1]-2):
            for k in range(npy.shape[2]-2):
                tmp = npy[i:i+3, j:j+3, k:k+3]
                if tmp.max()==0:
                    continue
                elif tmp[1,1,1]==0:
                    continue
                else:
                    t = 0
                    for id, kernel in enumerate(j_list):
                        if id==0 or id==3 or id==5 or id==7:
                            vt = vt0
                        elif id==2:
                            vt = vt2
                        else:
                            vt = vt1
                        
                        for n in range(2):
                            end_point = [x+i+1, y+j+1, z+k+1]                            
                            if n==1:
                                kernel = np.fliplr(kernel)
                            if (tmp==kernel).all():
                                if n==0:
                                    end_point.extend(vt[0][0])
                                if n==1:
                                    end_point.extend(vt[3][0])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel)).all():
                                if n==0:
                                    end_point.extend(vt[0][1])
                                if n==1:
                                    end_point.extend(vt[3][1])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 2)).all():
                                if n==0:
                                    end_point.extend(vt[0][2])
                                if n==1:
                                    end_point.extend(vt[3][2])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 3)).all():
                                if n==0:
                                    end_point.extend(vt[0][3])
                                if n==1:
                                    end_point.extend(vt[3][3])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 1, (1,2))).all():
                                if n==0:
                                    end_point.extend(vt[1][0])
                                if n==1:
                                    end_point.extend(vt[4][0])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 2, (1,2))).all():
                                if n==0:
                                    end_point.extend(vt[1][1])
                                if n==1:
                                    end_point.extend(vt[4][1])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 3, (1,2))).all():
                                if n==0:
                                    end_point.extend(vt[1][2])
                                if n==1:
                                    end_point.extend(vt[4][2])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 1, (2,0))).all():
                                if n==0:
                                    end_point.extend(vt[2][0])
                                if n==1:
                                    end_point.extend(vt[5][0])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 2, (2,0))).all():
                                if n==0:
                                    end_point.extend(vt[2][1])
                                if n==1:
                                    end_point.extend(vt[5][1])
                                end_point.append(label_id)
                                endlist.append(end_point)
                            elif (tmp==np.rot90(kernel, 3, (2,0))).all():
                                if n==0:
                                    end_point.extend(vt[2][2])
                                if n==1:
                                    end_point.extend(vt[5][2])
                                end_point.append(label_id)
                                endlist.append(end_point)
    endlist = get_unique_list(endlist)
    return endlist

def vec_table():
    vt0 = [[[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1]],
            [[1,1,-1],[-1,1,-1],[-1,1,1],[0,0,0]],
            [[-1,1,1],[-1,-1,1],[1,-1,1],[0,0,0]],
            [[1,1,-1],[1,1,1],[1,-1,1],[1,-1,-1]],
            [[-1,1,-1],[-1,1,1],[1,1,1],[0,0,0]],
            [[-1,1,-1],[-1,-1,-1],[1,1,-1],[0,0,0]]]

    vt1 = [[[0,1,1],[0,-1,1],[0,-1,-1],[0,1,-1]],
            [[1,1,0],[0,1,-1],[-1,1,0],[0,0,0]],
            [[-1,0,1],[0,-1,1],[1,0,1],[0,0,0]],
            [[0,1,-1],[0,1,1],[0,-1,1],[0,-1,-1]],
            [[-1,1,0],[0,1,1],[1,1,0],[0,0,0]],
            [[-1,0,-1],[0,-1,-1],[1,0,-1],[0,0,0]]]

    vt2 = [[[0,1,0],[0,0,1],[0,-1,0],[0,0,-1]],
            [[0,1,0],[0,1,0],[0,1,0],[0,0,0]],
            [[-1,0,0],[0,-1,0],[1,0,0],[0,0,0]],
            [[0,1,0],[0,0,1],[0,-1,0],[0,0,-1]],
            [[0,1,0],[0,1,0],[0,1,0],[0,0,0]],
            [[-1,0,0],[0,-1,0],[1,0,0],[0,0,0]]]

    return vt0, vt1, vt2

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

#make kernel
def judge_list():
    kernel = np.zeros((3,3,3))
    kernel[1,1,1]=1
    
    kernel0=copy.deepcopy(kernel)
    kernel0[0,0,0]=1

    kernel1=copy.deepcopy(kernel)
    kernel1[0,0,1]=1

    kernel2=copy.deepcopy(kernel)
    kernel2[0,1,1]=1

    kernel3=copy.deepcopy(kernel0)
    kernel3[0,1,1]=1

    kernel4=copy.deepcopy(kernel1)
    kernel4[0,1,1]=1

    kernel5=copy.deepcopy(kernel3)
    kernel5[0,0,1]=1

    kernel6=copy.deepcopy(kernel4)
    kernel6[1,0,1]=1

    kernel7=copy.deepcopy(kernel4)
    kernel7[0,0,0]=1
    kernel7[0,1,0]=1

    return [kernel0.astype('int'), kernel1.astype('int'), kernel2.astype('int'), kernel3.astype('int'), kernel4.astype('int'), kernel5.astype('int'), kernel6.astype('int'), kernel7.astype('int')]

#confirm binarize threshold
def binary_slice(npy_path, th):
    npy = np.load(npy_path)
    npy = np.where(npy>=th, 255, 0)
    save_path = Path('trace/test')
    save_path.mkdir(exist_ok=True)
    for i in range(npy.shape[2]):
        cv2.imwrite(f'{str(save_path)}/{i:03}.tif',npy[:,:,i])


# make denoise and skel image from binarized 3Dimage
def denoise_skel_image(npy, labels):
    denoised = np.zeros_like(npy)
    skel = np.zeros_like(npy)
    
    for i in tqdm(range(1, labels.max()+1)):
        tmp = npy[labels==i]
        if len(tmp)<30:
            continue
        tmp = np.copy(npy)
        tmp[labels!=i]=0
        denoised = np.maximum(denoised, tmp)
        sk_tmp = skeletonize(tmp)
        skel = np.maximum(skel, sk_tmp)
    
    return denoised, skel


if __name__ == '__main__':
    # pred_npy_path = 'Result/0807/test_pred.npy'
    # seed_th = 120
    # save_dir = Path('Result/0808/trace')
    # save_dir.mkdir(exist_ok = True)
    # main(save_dir, pred_npy_path, seed_th) #ep_list　の作成 denoised, skeled npy の作成


    dir_path = Path('Result/1012')
    dir_path.mkdir(exist_ok=True)

    traced = 'Result/0808/trace_s120_60/gaused_s120_60.npy'
    pred = 'Result/0807/test_pred.npy'
    end_points_slice(dir_path, traced, pred, 120)