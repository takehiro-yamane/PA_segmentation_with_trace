import trace
import ori_gaussian
import end_point_slice
from pathlib import Path
import numpy as np



pred_npy_path = 'Result/0807/test_pred.npy'
seed_th = 100
trace_th = 60
save_dir = Path(f'Result/0808/trace_s{seed_th}_{trace_th}')
save_dir.mkdir(exist_ok = True)

## 端点のcsvの作成、 細線化画像の作成
end_point_slice.main(save_dir, pred_npy_path, seed_th)


## トレース
skel_path = f'{str(save_dir)}/skeled.npy'
skeled = np.load(skel_path)
pred = np.load(pred_npy_path)
trace.main(save_dir, pred, skeled, trace_th, seed_th)

## 1voxel毎にガウシアンブラーをかける
traced_npy_path = f'{str(save_dir)}/traced_skel_s{seed_th}_{trace_th}.npy'
traced_npy = np.load(traced_npy_path)
ori_gaussian.main(save_dir, traced_npy, trace_th, seed_th)
