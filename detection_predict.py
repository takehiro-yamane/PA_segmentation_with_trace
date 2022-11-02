from datetime import datetime
from PIL import Image
from skimage import feature
import torch
import numpy as np
from pathlib import Path
import cv2
from networks import UNet
import argparse
from collections import OrderedDict
from tqdm import tqdm
from utils.load import VesselImageLoad
from other_codes import slice_mip

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-i",
        "--input_path",
        dest="input_path",
        help="dataset's path",
        default="./image/test",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        dest="output_path",
        help="output path",
        default="./output/detection",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="load weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", help="whether use CUDA", action="store_true", default="True"
    )

    args = parser.parse_args()
    return args


class Predict:
    def __init__(self, args):
        self.net = args.net
        self.gpu = args.gpu

        self.ori_path = args.input_path
        ori_paths = sorted(args.input_path.glob('*.tif'))
        
        data_loader = VesselImageLoad(ori_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=4, shuffle=False, num_workers=8, pin_memory=True
        )
        # self.number_of_traindata = data_loader.__len__()

        self.save_ori_path = args.output_path / Path("ori")
        self.save_pred_path = args.output_path / Path("pred")

        # self.save_ori_path.mkdir(parents=True, exist_ok=True)
        # self.save_pred_path.mkdir(parents=True, exist_ok=True)


    def pred(self, ori):
        img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1])
        )

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()
            mask_pred = self.net(img)

        pre_img = mask_pred.detach().cpu().numpy()[0, 0]
        
        pre_img = (pre_img * 255).astype(np.uint8)
        return pre_img

    #extract feature
    def extruct_f(self, ori):
        img = (ori.astype(np.float32) / ori.max()).reshape(
            (1, ori.shape[0], ori.shape[1])
        )

        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            if self.gpu:
                img = img.cuda()

            pre, mask_pred = self.net.forward2(img) 
        mask_pred = mask_pred.detach().cpu().numpy()
        mask_pred = np.squeeze(mask_pred, 0)
        # np.save('/home/kazuya/experiment_unet/pseudo_label/extract_feature//1129/feature', mask_pred)
        pre_img = pre.detach().cpu().numpy()[0, 0]
        pre_img = (pre_img * 255).astype(np.uint8)
        return mask_pred, pre_img


    def main(self):
        self.net.eval()
        # path def
        # ori_path = Path('/home/kazuya/WSISPDR_unet/image/test/ori')
        # pred_con_test = Path('/home/kazuya/experiment_unet/pred_con_test')
        
        ori_path = Path(args.input_path)
        pred_con_test = Path(args.output_path)

        pred_pre_path = pred_con_test/'pre'
        pred_ori_path = pred_con_test/'ori'
        pred_f_path = pred_con_test/'feature'
        pred_pre_path.mkdir(exist_ok=True)
        # pred_ori_path.mkdir(exist_ok=True)
        # pred_f_path.mkdir(exist_ok=True)

        # w/o dataloader version
        # paths = sorted(ori_path.glob("*.tif"))
        # for i, path in enumerate(tqdm(paths)):
        #     ori = np.array(Image.open(path))
        #     # pre_img = self.pred(ori)
        #     feature, pre_img = self.extruct_f(ori)
        #     # np.save(str(pred_f_path/'{:04}').format(i), feature)
        #     cv2.imwrite(str(pred_pre_path/'pre_{:04}.tif').format(i), pre_img)
        #     cv2.imwrite(str(pred_ori_path/'ori_{:04}.tif').format(i), ori)
        #     if i == 0:
        #         p = pre_img
        #         # o = ori
        #         p = np.expand_dims(p,2)
        #         # o = np.expand_dims(o,2)

        #     else:
        #         pre_img = np.expand_dims(pre_img, 2)
        #         # ori = np.expand_dims(ori, 2)
        #         p = np.concatenate([p, pre_img], 2)
        #         # o = np.concatenate([o, ori], 2)

        # np.save(str(pred_pre_path), p)
        # np.save(str(pred_ori_path), o)

        # dataloader version
        for id, data in enumerate(tqdm(self.train_dataset_loader)):
            img = data["image"]
            with torch.no_grad():
                if self.gpu:
                    img = img.cuda()
                # pre, feature = self.net.forward2(img) 
                pre = self.net(img) 
            pre = pre.detach().cpu().numpy()
            # feature = feature.detach().cpu().numpy()
            for j in range(pre.shape[0]):
                tmp = pre[j]
                tmp = np.squeeze(tmp)
                tmp = (tmp * 255).astype(np.uint8)
                cv2.imwrite(str(pred_pre_path/'pre_{:04}_{:02}.tif').format(id, j), tmp)

            # patch
            # con_save(0, pre, pred_pre_path, id)
            # con_save(1, pre, pred_pre_path, id)
            # con_save(2, pre, pred_pre_path, id)
            # con_save(3, pre, pred_pre_path, id)

        slice_mip.reconstruct(str(pred_con_test/'pre'), pred_pre_path)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def prediction(output_path, weight_path):
    args.output_path = Path(output_path)
    args.output_path.mkdir(exist_ok=True)
    args.weight_path = weight_path
    net = UNet(n_channels=1, n_classes=1)
    weight = torch.load(args.weight_path, map_location="cpu")
    weight = fix_model_state_dict(weight)
    # net.load_state_dict(torch.load(args.weight_path, map_location="cpu"))
    net.load_state_dict(weight)

    if args.gpu:
        net.cuda()
        # net = torch.nn.DataParallel(net)

    args.net = net

    pred = Predict(args)
    # pred = PredictFmeasure(args)

    pred.main()


if __name__ == "__main__":
    args = parse_args()
    args.input_path = Path('Result/0807/train/ori')

    prediction('Result/0809/train', 'Result/0809/weight')
    prediction('Result/0809-normal-pseudo/train', 'Result/0809-normal-pseudo/weight')
