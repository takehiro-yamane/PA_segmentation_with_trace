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
import copy


"""
ハイパラ
    trace threshold

必要な機能
    済
    閾値以上の領域を追跡
    輝度の高い領域を取得
    追跡中のベクトル
    他の端点に着くとトレース終了
    追跡方向を読み込む
    端点のリスト
    細線化にトレース結果を追加していく
    
    未
    ガウシアンをかける

その他
    三次元画像全体から探索
    初手は読み込んだ追跡方向から

"""

def trace(pred, ep_list, coordinate_list, th):
    ret_coor_list = []
    for n, ep in enumerate(tqdm(ep_list)):
        if ep[7]==1:
            continue
        x = ep[0]
        y = ep[1]
        z = ep[2]
        coor = [x,y,z]
        vec = [ep[3],ep[4],ep[5]]

        trace_coor_list = []
        while(1):
            slice_plane = best_slice(vec)
            vec = vec_arrange(vec)
            if slice_plane==0:
                next_coor, next_vec = search(0, vec, coor, pred, th)
            elif slice_plane==1:
                next_coor, next_vec = search(1, vec, coor, pred, th)
            else:
                next_coor, next_vec = search(2, vec, coor, pred, th)

            # 次が見つからなければ終了
            if next_vec[0] == -20:
                break

            # next_coor が他の端点であるか判定
            idx = judge_endpoint(next_coor, coordinate_list, n)
            if idx != -1:
                ep_list[idx][7] = 1
                break
            
            if judge_loop(trace_coor_list, next_coor):
                break
            
            trace_coor_list.append(next_coor)
            vec = next_vec
            coor = next_coor
        
        ret_coor_list.extend(trace_coor_list)
    return ret_coor_list


def trace_debug(pred, ep_list, coordinate_list, th, debug_save_dir):
    ret_coor_list = []
    
    for n, ep in enumerate(tqdm(ep_list)):
        save_path = debug_save_dir/f'{n:04}'
        save_path.mkdir(exist_ok=True)
        s_xy = save_path/'xy'
        s_yz = save_path/'yz'
        s_zx = save_path/'zx'
        s_xy.mkdir(exist_ok=True)
        s_yz.mkdir(exist_ok=True)
        s_zx.mkdir(exist_ok=True)

        if ep[7]==1:
            continue
        x = ep[0]
        y = ep[1]
        z = ep[2]
        coor = [x,y,z]
        vec = [ep[3],ep[4],ep[5]]

        trace_coor_list = []
        counter = 0
        slice_change_flg = 0
        while(1):
            
            slice_plane = best_slice(vec)
            vec = vec_arrange(vec)
            if slice_plane==0:
                next_coor, next_vec = search(0, vec, coor, pred, th)
            elif slice_plane==1:
                next_coor, next_vec = search(1, vec, coor, pred, th)
            else:
                next_coor, next_vec = search(2, vec, coor, pred, th)

            # 次が見つからなければ終了
            if next_vec[0] == -20:
                slice_plane = next_slice(slice_plane)
                slice_change_flg += 1

                if slice_change_flg>=2:
                    with open(f'{save_path}/trace_log_{n:04}', 'a') as f:
                        f.write('トレース先がみつからないため終了 \n')
                    break
                
                else:
                    continue

            # next_coor が他の端点であるか判定
            idx = judge_endpoint(next_coor, coordinate_list, n)
            if idx != -1:
                ep_list[idx][7] = 1
                with open(f'{save_path}/trace_log_{n:04}', 'a') as f:
                    f.write('次のトレース先が他の端点の周囲3voxelsに含まれているためトレース終了 \n')
                break
            
            if judge_loop(trace_coor_list, next_coor):
                with open(f'{save_path}/trace_log_{n:04}', 'a') as f:
                    f.write('トレースがループしたので終了 \n')
                break

            with open(f'{save_path}/trace_log_{n:04}', 'a') as f:
                f.write(f'{counter:02} : slice plane={vec}, c={coor}, int={pred[coor[0],coor[1],coor[2]]} \n')

            img_save_debug(s_xy, s_yz, s_zx, counter, coor, pred)

            trace_coor_list.append(next_coor)
            vec = next_vec
            coor = next_coor

            counter = counter+1
            slice_change_flg = 0

        img_save_add(s_xy, s_yz, s_zx, counter, coor, pred)

        ret_coor_list.extend(trace_coor_list)
    return ret_coor_list


def img_save_debug(s_xy, s_yz, s_zx, counter, coor, ori):
    x = coor[0]
    y = coor[1]
    z = coor[2]
    x_min = x-50
    x_max = x+50
    y_min = y-50
    y_max = y+50
    z_min = z-50
    z_max = z+50
    
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

    xy = ori[x_min:x_max, y_min:y_max, z]
    yz = ori[x, y_min:y_max, z_min:z_max]
    zx = ori[x_min:x_max, y, z_min:z_max]

    cv2.imwrite(f'{s_xy}/{counter:02}.tif', xy)
    cv2.imwrite(f'{s_yz}/{counter:02}.tif', yz)
    cv2.imwrite(f'{s_zx}/{counter:02}.tif', zx)


def img_save_add(s_xy, s_yz, s_zx, counter, coor, ori):
    x = coor[0]
    y = coor[1]
    z = coor[2]
    x_min = x-50
    x_max = x+50
    y_min = y-50
    y_max = y+50
    z_min = z-50
    z_max = z+50
    
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

    for i in range(3):
        counter += 1
        z += 1
        if z>=1120:
            break
        xy = ori[x_min:x_max, y_min:y_max, z]
        cv2.imwrite(f'{s_xy}/{counter:02}.tif', xy)
    counter -= 3

    for i in range(3):
        counter += 1
        x += 1
        if x>=1120:
            break
        yz = ori[x, y_min:y_max, z_min:z_max]
        cv2.imwrite(f'{s_yz}/{counter:02}.tif', yz)
    counter -= 3
    
    for i in range(3):
        counter += 1
        y += 1
        if y>=1120:
            break
        zx = ori[x_min:x_max, y, z_min:z_max]
        cv2.imwrite(f'{s_zx}/{counter:02}.tif', zx)
    counter -= 3


def judge_loop(trace_list, next_coor):
    if next_coor in trace_list:
        return True
    else:
        return False

def vec_arrange(vec):
    if vec[0]>0:
        vec[0] = 1
    if vec[1]>0:
        vec[1] = 1
    if vec[2]>0:
        vec[2] = 1
    if vec[0]<0:
        vec[0] = -1
    if vec[1]<0:
        vec[1] = -1
    if vec[2]<0:
        vec[2] = -1
    return vec


# 次のスライスにおけるトレース先の座標を返す
# base_slice : 現在の端点が含まれるスライス
# slice_plane : 次にスライスする平面
# vec : 一つ前と現在の端点間のベクトル
# coor : 現在の端点の座標
# pred : 3D推定画像
def search(slice_plane, vec, coor, pred, th):
    if slice_plane == 0:
        x = coor[0]+vec[0]
        if x<0 or x>=pred.shape[0]:
            return [-1,-1,-1], [-20,-1,-1]
        next_slice = pred[x,:,:]
        c0, c1, ret_vec = search_0(vec, coor, next_slice, th)
        return [x, c0, c1], ret_vec

    elif slice_plane == 1:
        y = coor[1]+vec[1]
        if y<0 or y>=pred.shape[1]:
            return [-1,-1,-1], [-20,-1,-1]
        next_slice = pred[:,y,:]
        c0, c1, ret_vec = search_1(vec, coor, next_slice, th)
        return [c0, y, c1], ret_vec

    else:
        z = coor[2]+vec[2]
        if z<0 or z>=pred.shape[2]:
            return [-1,-1,-1], [-20,-1,-1]
        next_slice = pred[:,:,z]
        c0, c1, ret_vec = search_2(vec, coor, next_slice, th)
        return [c0, c1, z], ret_vec


def search_0(vec, coor, next_slice, th):
    pre_ep = next_slice[coor[1], coor[2]]
    if pre_ep>=60:
        #輝度が最大の点を求める
        c0, c1 = max_intensity(next_slice, coor[1], coor[2])
    else:
        #範囲を広げる
        c0, c1 = surround(next_slice, coor[1], coor[2], th)
    
    if c0 == -1:
        return -1,-1, [-20,-1,-1]
    
    ret_vec = calc_vec(vec, coor, c0, c1, 0)

    return c0, c1, ret_vec

def search_1(vec, coor, next_slice, th):
    pre_ep = next_slice[coor[0], coor[2]]
    if pre_ep>=60:
        #輝度が最大の点を求める
        c0, c1 = max_intensity(next_slice, coor[0], coor[2])
    else:
        #範囲を広げる
        c0, c1 = surround(next_slice, coor[0], coor[2], th)
    
    if c0 == -1:
        return -1,-1, [-20,-1,-1]
    
    ret_vec = calc_vec(vec, coor, c0, c1, 1)

    return c0, c1, ret_vec

def search_2(vec, coor, next_slice, th):
    pre_ep = next_slice[coor[0], coor[1]]
    if pre_ep>=60:
        #輝度が最大の点を求める
        c0, c1 = max_intensity(next_slice, coor[0], coor[1])
    else:
        #範囲を広げる
        c0, c1 = surround(next_slice, coor[0], coor[1], th)
    
    if c0 == -1:
        return -1,-1,[-20,-1,-1]
    
    ret_vec = calc_vec(vec, coor, c0, c1, 2)

    return c0, c1, ret_vec

#輝度が最大の点を求める
def max_intensity(slice, c0, c1):
    tmp0 = c0
    tmp1 = c1
    while(1):
        a_min = tmp0-1
        a_max = tmp0+2
        b_min = tmp1-1
        b_max = tmp1+2
        if a_min<0 or b_min<0 or a_max > slice.shape[0] or b_max > slice.shape[1]:
            return tmp0, tmp1

        kernel = slice[a_min:a_max, b_min:b_max]

        kernel_max = np.max(kernel)
        max_coor = np.where(kernel==kernel_max)
        max_coor_list = npw2list(max_coor)

        if [1,1] in max_coor_list:
            return tmp0, tmp1

        tmp0 += max_coor[0][0]-1
        tmp1 += max_coor[1][0]-1

def npw2list(ret):
    ret_list = []
    for i in range(len(ret[0])):
        tmp = [ret[0][i], ret[1][i]]
        ret_list.append(tmp)
    return ret_list

# 探索範囲を広げる
def surround(slice, c0, c1, th):
    for i in range(1,6):
        a_min = c0-i
        a_max = c0+i+1
        b_min = c1-i
        b_max = c1+i+1
        if a_min<0 or b_min<0 or a_max>=slice.shape[0] or b_max>=slice.shape[1]:
            continue
            
        tmp_kernel = slice[a_min:a_max, b_min:b_max]
        tmp_max = np.max(tmp_kernel)
        if tmp_max>=th:
            max_coor = np.where(tmp_kernel==tmp_max)
            tmp_x = c0 + max_coor[0][0]-i
            tmp_y = c1 + max_coor[1][0]-i
            x, y = max_intensity(slice, tmp_x, tmp_y)
            return x, y
    
    return -1, -1

# ベクトル計算
def calc_vec(vec, coor, c0, c1, slice_plane):
    if slice_plane == 0:
        post_coor = [coor[0]+vec[0],c0,c1]
    elif slice_plane == 1:
        post_coor = [c0, coor[1]+vec[1], c1]
    else:
        post_coor = [c0, c1, coor[2]+vec[2]]
    
    post_coor = np.array(post_coor)
    coor = np.array(coor)
    vec = post_coor - coor
    return vec

# 端点判定
# 探索中の端点を除く
def judge_endpoint(new_coor, epc_list, n):
    judge_list = np.arange(7)
    judge_list -= 3

    c_list = copy.deepcopy(epc_list)
    del c_list[n]

    for i in judge_list:
        for j in judge_list:
            for k in judge_list:
                tmp_coor = [new_coor[0]+i, new_coor[1]+j, new_coor[2]+k]
                if tmp_coor in c_list:
                    idx = c_list.index(tmp_coor)
                    return idx
    return -1


## xy = 2, yz = 0, zx = 1
def best_slice(vec):
    x = abs(vec[0])
    y = abs(vec[1])
    z = abs(vec[2])
    max_vec = max(x, y, z)
    if x==max_vec:
        return 0
    elif y==max_vec:
        return 1
    else:
        return 2

# スライスの方向を変更する
def next_slice(n):
    ret = n+1
    if ret==3:
        ret = 0
    return ret


# 細線化にトレース結果を追加

def plot_trace(seed, coor_list):
    tmp = np.copy(seed)
    for coor in coor_list:
        tmp[coor[0],coor[1],coor[2]]=255
    return tmp

# slice
def slice_3d(save_path, npy):
    for i in range(npy.shape[2]):
        cv2.imwrite(f'{str(save_path)}/{i:03}.tif', npy[:,:,i])

def main(save_dir, pred, skeled, th, seed_th):
    seed = np.where(pred>=seed_th, 255, 0)
    skeled = np.where(skeled>0, 255, 0)

    df = pd.read_csv(f'{str(save_dir)}/trace_s120_60/ep_list.csv')
    df = df.drop(df.columns[[0]], axis=1)
    df_ep = df.drop(df.columns[[3,4,5,6]], axis=1)
    df[7] = 0
    ep_list = df.values.tolist()
    epc_list = df_ep.values.tolist()
    
    # coor_list = trace(pred, ep_list, epc_list, th)
    coor_list = trace_debug(pred, ep_list, epc_list, th, Path('Result/1101/test'))

    # traced = plot_trace(skeled, coor_list)
    # np.save(f'{str(save_dir)}/traced_skel_s{seed_th}_{th}.npy', traced)
    # save = save_dir/f'traced_skel_s{seed_th}_{th}'
    # save.mkdir(exist_ok=True)
    # slice_3d(save, traced)

if __name__ == '__main__':
    skel_path = 'Result/0808/trace_s120_60/skeled.npy'
    skeled = np.load(skel_path)
    pred_path = 'Result/0807/test_pred.npy'
    pred = np.load(pred_path)
    save_dir = Path('Result/0808')

    th = 60
    main(save_dir, pred, skeled, th, 120)
    # th = 50
    # main(save_dir, pred, skeled, th)
    # th = 40
    # main(save_dir, pred, skeled, th)
    # th = 30
    # main(save_dir, pred, skeled, th)
    # th = 20
    # main(save_dir, pred, skeled, th, seed_th=120)
    # th = 10
    # main(save_dir, pred, skeled, th, seed_th=120)