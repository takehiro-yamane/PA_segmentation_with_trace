from numpy.matrixlib.defmatrix import _convert_from_string
from tqdm import tqdm
from torch import optim
import torch.utils.data
import torch.nn as nn
from utils.load_mask import CellImageLoad
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from networks import UNet
# from networks import VNet
import argparse
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument(
        "-t",
        "--train_path",
        dest="train_path",
        help="training dataset's path",
        default="./image/train",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--val_path",
        dest="val_path",
        help="validation data path",
        default="./image/val",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask_path",
        dest="mask_path",
        help="lossmask data path",
        default="./image/mask",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--weight_path",
        dest="weight_path",
        help="save weight path",
        default="./weight/best.pth",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", default=True, help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", help="batch_size", default=80, type=int
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", help="epochs", default=500, type=int
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="learning late",
        default=1e-3,
        type=float,
    )

    args = parser.parse_args()
    return args


class _TrainBase:
    def __init__(self, args):
        ori_paths = self.gather_path(args.train_path, "ori")
        gt_paths = self.gather_path(args.train_path, "gt")
        mask_paths = self.gather_path(args.train_path, "mask")
        data_loader = CellImageLoad(ori_paths, gt_paths, mask_paths)
        self.train_dataset_loader = torch.utils.data.DataLoader(
            data_loader, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        self.number_of_traindata = data_loader.__len__()

        self.save_weight_path = args.weight_path
        self.save_weight_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_weight_path.parent.joinpath("epoch_weight").mkdir(
            parents=True, exist_ok=True
        )
        print(
            "Starting training:\nEpochs: {}\nBatch size: {} \nLearning rate: {}\ngpu:{}\n".format(
                args.epochs, args.batch_size, args.learning_rate, args.gpu
            )
        )

        self.net = net

        self.train = None
        self.val = None

        self.N_train = None
        self.optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu = args.gpu
        #self.criterion = nn.MSELoss()
        self.losses = []
        self.val_losses = []
        self.evals = []
        self.epoch_loss = 0
        self.bad = 0

    def gather_path(self, train_paths, mode):
        ori_paths = []
        for train_path in train_paths:
            ori_paths.extend(sorted(train_path.joinpath(mode).glob("*.*")))
        return ori_paths

    def maskedmseloss(self, input, target, mask, epoch):
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                    "This will likely lead to incorrect results due to broadcasting. "
                    "Please ensure they have the same size.".format(target.size(), input.size()),
                    stacklevel=2)
        ret = (input - target) ** 2
        ret = ret[mask == 0]
        
        # ret2 = ret[target>=0.1]
        # ret2 = torch.mean(ret2)
        # ret2 = ret2 / 2
        

        ret = torch.mean(ret)

        # ret = ret + ret2
        writer.add_scalar('loss', ret, epoch)
        return ret
    
    def weight_mseloss(self, input, target, mask, epoch, w, th):
        if not (target.size() == input.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                    "This will likely lead to incorrect results due to broadcasting. "
                    "Please ensure they have the same size.".format(target.size(), input.size()),
                    stacklevel=2)
        
        
        tmp = (input - target) ** 2
        f = tmp[target>th]
        f = torch.mean(f)
        b = tmp[mask <= th]
        b = torch.mean(b)
        ret = b + (f*w)
        writer.add_scalar('loss', ret, epoch)
        return ret   

class TrainNet(_TrainBase):
    def loss_calculate(self, masks_probs_flat, true_masks_flat, lossmask, epoch):
        #return self.criterion(masks_probs_flat, true_masks_flat, loss_mask)
        return self.maskedmseloss(masks_probs_flat, true_masks_flat, lossmask, epoch)
        # return self.weigted_mseloss(masks_probs_flat, true_masks_flat, lossmask, epoch, 2, 0)

    def main(self):
        iteration = 0
        for epoch in tqdm(range(self.epochs)):
            print("Starting epoch {}/{}.".format(epoch + 1, self.epochs))
            self.net.train()

            for i, data in enumerate(self.train_dataset_loader):
                imgs = data["image"]
                true_masks = data["gt"]
                loss_masks = data["mask"]

                if self.gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()
                    loss_masks = loss_masks.cuda()

                masks_pred = self.net(imgs)
                loss = self.loss_calculate(masks_pred, true_masks, loss_masks, epoch)
                self.epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            iteration += 1

        torch.save(self.net.state_dict(), str(self.save_weight_path))


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    writer = SummaryWriter()
    args = parse_args()
    args.gpu = True
    args.train_path = "Result/0807/train"
    args.train_path = [Path(args.train_path)]
    args.val_path = [Path(args.val_path)]
    # save weight path
    args.weight_path = 'Result/0807/weight'
    args.weight_path = Path(args.weight_path)
    args.epochs = 1001

    # define model
    net = UNet(n_channels=1, n_classes=1)
    # weight = torch.load(args.weight_path, map_location="cpu")
    # weight = fix_model_state_dict(weight)
    # net.load_state_dict(weight)

    if args.gpu:
        net.cuda()
        net = torch.nn.DataParallel(net)

    args.net = net
    train = TrainNet(args)
    
    train.main()
    writer.close()

