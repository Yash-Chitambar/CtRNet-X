import sys
import os
base_dir = os.path.abspath("../")
sys.path.append(base_dir)

import multiprocessing as mp
import time

from PIL import Image

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import glob
import kornia
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from imageloaders.DREAM import ImageDataLoaderSynthetic, LabelGenerator

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from utils import *
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()

args = parser.parse_args("")

args.n_kp = 12
args.scale = 0.5
args.height = 480
args.width = 640
args.fx = -320.
args.fy = -320.
args.px = 320.
args.py = 240.
args.lim=[-1., 1., -1., 1.]
args.base_dir = '/CtRNet-robot-pose-estimation'

args.data_folder = ['DREAM_ds/panda_synth_train_dr', 'DREAM_ds/panda_synth_test_photo', 'DREAM_ds/panda_synth_test_dr/panda_synth_test_dr']
args.test_data_folder = 'DREAM_ds/panda_synth_test_photo'
args.use_gpu = True
args.batch_size = 64
args.num_workers = 8
args.lr = 1e-5
args.beta1 = 0.9
args.n_epoch = 1000
args.out_dir = 'outputs'
args.ckp_per_epoch = 10


args.height = int(args.height * args.scale)
args.width = int(args.width * args.scale)
args.fx = args.fx * args.scale
args.fy = args.fy * args.scale
args.px = args.px * args.scale
args.py = args.py * args.scale

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train','valid']:
    # datasets[phase] = ImageDataLoaderSynthetic(data_folder = args.data_folder if phase=='train' else args.test_data_folder, scale = args.scale, trans_to_tensor = trans_to_tensor)
    if phase=='train':
        datasets[phase] = ConcatDataset([ImageDataLoaderSynthetic(data_folder = train_dir, scale = args.scale, trans_to_tensor = trans_to_tensor) for train_dir in args.data_folder ])
    else:
        datasets[phase] = ImageDataLoaderSynthetic(data_folder = args.test_data_folder, scale = args.scale, trans_to_tensor = trans_to_tensor)

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])
    
for train_dir in args.data_folder:
    labelgenerator = LabelGenerator(args, train_dir)
    print("Generating ground-truth masks and keypoints for {}".format(train_dir))
    for i in tqdm(range(len(labelgenerator.ndds_dataset))):
        labelgenerator.generate_mask(i)
        labelgenerator.generate_keypoints(i)

print("Generating ground-truth masks and keypoints for {}".format(args.test_data_folder))
labelgenerator = LabelGenerator(args, args.test_data_folder)
for i in tqdm(range(len(labelgenerator.ndds_dataset))):
    labelgenerator.generate_mask(i)
    labelgenerator.generate_keypoints(i)

import kornia as kn

from models.keypoint_seg_resnet import KeyPointSegNet

keypoint_seg_predictor = KeyPointSegNet(args, use_gpu=args.use_gpu)
if args.use_gpu:
    keypoint_seg_predictor = keypoint_seg_predictor.cuda()

keypoint_seg_predictor = torch.nn.DataParallel(keypoint_seg_predictor, device_ids=[0])


from models.heatmap import GaussianHeatmapLoss, heatmap_to_keypoints
heatmapLoss = GaussianHeatmapLoss()
criterionBCE = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(keypoint_seg_predictor.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

if args.use_gpu:
    heatmapLoss = heatmapLoss.cuda()
    criterionBCE = criterionBCE.cuda()

if args.use_gpu:
    device = "cuda"
else:
    device = "cpu"
    
start_epoch = 0



best_valid_loss = np.inf

checkpoint_path = os.path.join(args.out_dir, 'net_best.pth')
if os.path.exists(checkpoint_path):
    print("Loading checkpoint from '{}'...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    keypoint_seg_predictor.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_valid_loss = checkpoint['best_valid_loss']
    print("Checkpoint loaded: starting from epoch {}, best validation loss: {}".format(start_epoch, best_valid_loss))

epoch_writer = SummaryWriter(comment="_writter")

for epoch in range(start_epoch, args.n_epoch):
    phases = ['train','valid']

    for phase in phases:
        iter_writer = SummaryWriter(comment="_epoch_" + str(epoch) + "_" + phase)

        # train keypoint detector
        
        keypoint_seg_predictor.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_kp = AverageMeter()
        meter_loss_seg = AverageMeter()


        loader = dataloaders[phase]

        #bar = ProgressBar(maxval=data_n_batches[phase])
        for i, data in tqdm(enumerate(loader), total=data_n_batches[phase]):

            if args.use_gpu:
                if isinstance(data, list):
                    data = [d.cuda() for d in data]
                else:
                    data = data.cuda()

            # load data
            img, _, _, points_2d_gt_batch, mask_batch = data

            valid_point_table = torch.logical_and(torch.logical_and(points_2d_gt_batch[:,:,0] < args.width, points_2d_gt_batch[:,:,0] > 0), 
                                      torch.logical_and(points_2d_gt_batch[:,:,1] < args.height, points_2d_gt_batch[:,:,1] > 0))


            with torch.set_grad_enabled(phase == 'train'):


                # detect 2d keypoints
                heatmap, segmentation = keypoint_seg_predictor(img)

                loss_seg = criterionBCE(segmentation.squeeze(), mask_batch)
                loss_heatmap,_ = heatmapLoss(heatmap, points_2d_gt_batch)
                # scale heatmap loss 1000
                loss_heatmap = loss_heatmap * (1e+3)
                loss = loss_heatmap + loss_seg
                # loss = loss_heatmap

                meter_loss.update(loss.item(), n=img.size(0))
                meter_loss_kp.update(loss_heatmap.item(), n=img.size(0))
                meter_loss_seg.update(loss_seg.item(), n=img.size(0))

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(keypoint_seg_predictor.parameters(), 10)
                optimizer.step()

            # write to log

            iter_writer.add_scalar('loss_kp', loss_heatmap.item(), i)
            iter_writer.add_scalar('loss_seg', loss_seg.item(), i)
            iter_writer.add_scalar('loss_all', loss.item(), i)

            if (i%200==0 and phase=='train') or (phase=='valid' and i%20==0):

                points_2d = heatmap_to_keypoints(heatmap)
                img_np = to_numpy_img(img[0])
                img_np_pred = overwrite_image(img_np.copy(),points_2d[0].detach().cpu().numpy().squeeze().astype(int), color=(0,1,0),point_size=6)
                img_np_gt = overwrite_image(img_np.copy(),points_2d_gt_batch[0].detach().cpu().numpy().squeeze().astype(int), color=(0,1,0),point_size=6)
                iter_writer.add_image('[keypoint] gt vs predict', np.concatenate((img_np_gt,img_np_pred),axis=1), i, dataformats='HWC')

                iter_writer.add_image('[segmentation] gt vs predict', np.concatenate((mask_batch[0].squeeze().cpu().detach().numpy(),
                                                                        torch.sigmoid(segmentation[0]).squeeze().cpu().detach().numpy()),
                                                                        axis=1), i, dataformats='HW')

        log = '%s [%d/%d] Loss: %.6f, LR: %f' % (
            phase, epoch, args.n_epoch,
            meter_loss.avg,
            get_lr(optimizer))

        iter_writer.close()

        print(log)

        if phase == 'valid':
            epoch_writer.add_scalar('loss_kp_val', meter_loss_kp.avg, epoch)
            epoch_writer.add_scalar('loss_seg_val', meter_loss_seg.avg, epoch)
            epoch_writer.add_scalar('loss_all_val', meter_loss.avg, epoch)

            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg

                # torch.save(keypoint_seg_predictor.state_dict(), '%s/net_best.pth' % (args.out_dir))
                torch.save({'epoch': epoch,
                            'model_state_dict': keypoint_seg_predictor.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_valid_loss': best_valid_loss}, 
                           '%s/net_best.pth' % (args.out_dir))

            log = 'Best valid: %.6f' % (best_valid_loss)
            print(log)
            
            #torch.save(keypoint_seg_predictor.state_dict(), '%s/net_last.pth' % (args.out_dir))
            if epoch % args.ckp_per_epoch == 0:
                # torch.save(keypoint_seg_predictor.state_dict(), '%s/net_%d.pth' % (args.out_dir, epoch))
                torch.save({'epoch': epoch,
                    'model_state_dict': keypoint_seg_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_valid_loss': best_valid_loss}, 
                    '%s/net_%d.pth' % (args.out_dir, epoch))
        else:
            epoch_writer.add_scalar('loss_kp', meter_loss_kp.avg, epoch)
            epoch_writer.add_scalar('loss_seg', meter_loss_seg.avg, epoch)
            epoch_writer.add_scalar('loss_all', meter_loss.avg, epoch)
            
epoch_writer.close()






