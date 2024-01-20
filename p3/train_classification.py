"""
Author: Benny
Date: Nov 2019
"""
import math
import os
import sys

import torch
import numpy as np

import datetime
import logging

import yaml
from torch import nn
import provider
import argparse

from pathlib import Path
from timm.scheduler import CosineLRScheduler

from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'PromptModels'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default="2", help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='structure', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=5e-3, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=5e-2, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
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
    for j, (points, target) in enumerate(loader):

        if not args.use_cpu:
            points, target = points.to('cuda'), target.to('cuda')

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

    '''HYPER PARAMETER'''
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
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/ModelNet/modelnet40_normal_resampled/'
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=6, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=6)

    '''MODEL LOADING'''
    num_class = args.num_category
    from PromptModels.pointFormer import pointFormer
    model = pointFormer(num_classes=40, base_model='vit_base_patch16_224_in21k', in_channel=12)
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.3)

    if not args.use_cpu:
        model = model.to('cuda')
        criterion = criterion.to('cuda')

    log_string('No existing model, starting training from scratch...')
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

    # # check backwarding tokens
    # print("===> load parameter")
    # for name, param in model.named_parameters():
    #     print(f"{name},{param.shape}")
    # print("===> hot")
    # for name, param in model.named_parameters():
    #     if param.requires_grad is True:
    #         print(f"{name},{param.shape}")

    # trainable_pytorch_total_params = sum(p.numel() for n,p in model.named_parameters() if p.requires_grad)
    # print(trainable_pytorch_total_params)

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
        for batch_id, (points, target) in enumerate(trainDataLoader, 0):

            optimizer.zero_grad()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)

            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

            points = torch.Tensor(points)

            if not args.use_cpu:
                points, target = points.to('cuda'),target.to('cuda')

            # b,40
            pred = model(points)

            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1] # B
            correct = pred_choice.eq(target.long().data).cpu().sum() # B
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
