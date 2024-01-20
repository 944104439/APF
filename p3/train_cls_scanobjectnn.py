"""
Author: Benny
Date: Nov 2019
"""
import logging
import os
import sys

import torch
import numpy as np

import datetime

import shutil
import argparse

from pathlib import Path
from timm.scheduler import CosineLRScheduler
from torch import nn
from torchvision.transforms import transforms

from PromptModels.pointFormer import pointFormer
from data_utils.scanobjectnnhardest import ScanObjectNNHardest

# from data_utils.scanobjectnnhardest import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'PromptModels'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default="4,3", help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_category', default=15, type=int, help='training on scanobjnn')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--decay_rate', type=float, default=5e-2, help='decay rate')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def setup_seed(seed):  # setting up the random seed
    import random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test(model, loader, num_class=40):
    mean_correct = []
    # num_class 行，3列
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    classifier = classifier.to('cuda')
    for j, data in enumerate(loader):
        points = data['x']
        target = data['y']
        if not args.use_cpu:
            points, target = points.to('cuda'), target.to('cuda')
        points = points.contiguous()
        target = target.contiguous()
        # b,40
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    setup_seed(1463)

    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' %log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    data_path = '/home/ubuntu/h5_files/main_split/'

    from data_utils.transforms import PointsToTensor, PointCloudScaling, PointCloudCenterAndNormalize, PointCloudRotation
    train_transforms = transforms.Compose([
        PointsToTensor(),
        PointCloudScaling(scale=[0.9, 1.1]),
        PointCloudCenterAndNormalize(gravity_dim=1),
        PointCloudRotation(angle=[0.0, 1.0, 0.0]),
        # ChromaticDropGPU()
    ])
    val_transforms = transforms.Compose([
        PointsToTensor(),
        PointCloudCenterAndNormalize(gravity_dim=1)
    ])
    TRAIN_DATASET = ScanObjectNNHardest(num_points=2048, split='train', transform=train_transforms, data_dir=data_path)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=32, shuffle=True,
                                               num_workers=6, drop_last=True)
    TEST_DATASET = ScanObjectNNHardest(num_points=1024, split='test', transform=val_transforms, data_dir=data_path)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=32, shuffle=False,
                                              num_workers=6)

    '''MODEL LOADING'''
    num_class = args.num_category

    model = pointFormer(num_classes=15, base_model='vit_base_patch16_224_in21k', in_channel=4*2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.3)
    model = nn.DataParallel(model)

    if not args.use_cpu:
        model = model.to('cuda')
        criterion = criterion.to('cuda')

    start_epoch = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )

    scheduler = CosineLRScheduler(optimizer, t_initial=args.epoch, warmup_t=10,
                                  warmup_lr_init=1e-3,
                                  cycle_decay=0.05)  # , cycle_decay=0.1  LMK:cycle_decay->delay_rate

    # check backwarding tokens
    # print("===> load parameter")
    # for name, param in model.named_parameters():
    #     print(f"{name},{param.shape}")
    # print("===> HOT")
    # for name, param in model.named_parameters():
    #     if param.requires_grad is True:
    #         print(f"{name},{param.shape}")

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        # 训练模式
        model = model.train()

        scheduler.step(epoch)
        import time
        t1 = time.time()
        for batch_id, data in enumerate(trainDataLoader, 0):

            optimizer.zero_grad()
            points = data['x']
            target = data['y']
            # print(points.shape)
            if not args.use_cpu:
                points, target = points.to('cuda'), target.to('cuda')
            points = points.contiguous()
            target = target.contiguous()
            # b,40
            pred = model(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()# B
            mean_correct.append(correct.item() / float(points.shape[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
        # t2 = time.time()
        log_string(f"one epoch {time.time() - t1:.2f}s")
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(model.eval(), testDataLoader, num_class=num_class)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f (epoch: %d)' % (best_instance_acc, best_epoch))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
