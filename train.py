import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import spiking_resnet
from resnet19 import ResNet19

from functions import seed_all, data_transforms

import torchvision
import torchvision.datasets as datasets
from spikingjelly.activation_based.functional import reset_net

# This code is based on https://github.com/brain-intelligence-lab/temporal_efficient_training/blob/main/main_training_distribute.py

parser = argparse.ArgumentParser(description='PyTorch MP-Init + ta-SrGD Training')
parser.add_argument('-j',
                    '--workers',
                    default=2,
                    type=int,
                    metavar='N',
                    help='number of data loading workers per gpu (default: 2)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default: 0.1)',
                    dest='lr')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed',
                    default=1234,
                    type=int,
                    help='seed for initializing training. (default: 1234)')
parser.add_argument('--T',
                    default=4,
                    type=int,
                    metavar='N',
                    help='snn training timestep (default: 4)')
parser.add_argument('--init_tau',
                    default=2.0,
                    type=float,
                    metavar='N',
                    help='inital tau (default: 2.0)')
parser.add_argument('--init_thr',
                    default=2.0,
                    type=float,
                    metavar='N',
                    help='inital threshold (default: 2.0)')
parser.add_argument('--weight_decay',
                    default=5e-4,
                    type=float,
                    metavar='N',
                    help='weight decay of an optimizer (default: 4e-5)')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='N',
                    help='momentum of an optimizer (default: 0.9)')
parser.add_argument('--model', 
                    type=str,  
                    default="resnet19")
parser.add_argument('--dataset', 
                    type=str,  
                    default="cifar100")
parser.add_argument('--dataset_folder', 
                    type=str,  
                    default="/SSD")
parser.add_argument('--load_names', 
                    type=str,  
                    default=None)
parser.add_argument('--save_names', 
                    type=str,  
                    default=None)
args = parser.parse_args()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def main():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    print(args)
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:12345',
                            world_size=args.nprocs,
                            rank=local_rank)
    load_names = args.load_names
    save_names = args.save_names

    if args.model == "spiking":
        model = spiking_resnet.spiking_resnet34(T=args.T, init_tau=args.init_tau, init_thr=args.init_thr)
    elif args.model == "resnet19":
        model = ResNet19(T=args.T, init_tau=args.init_tau, init_thr=args.init_thr)
    else:
        raise ValueError(args.model)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    cudnn.benchmark = True

    train_transform, valid_transform = data_transforms(args.dataset)
    if args.dataset == "imagenet":
        # Data loading code
        train_dataset = datasets.ImageFolder(os.path.join(args.dataset_folder, 'ILSVRC2012/train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.dataset_folder, 'ILSVRC2012/val'), transform=valid_transform)
    elif args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(os.path.join(args.dataset_folder, "CIFAR10"), train=True,
                                                download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(os.path.join(args.dataset_folder, "CIFAR10"), train=False,
                                              download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(os.path.join(args.dataset_folder, "CIFAR100"), train=True,
                                                download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(os.path.join(args.dataset_folder, "CIFAR100"), train=False,
                                              download=True, transform=valid_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=val_sampler)
    
    # for evaluation of a snapshot
    if args.evaluate:
        snapshot = torch.load(os.path.join("./snapshots", load_names + ".pth.tar"),
                                map_location=torch.device('cuda:{}'.format(local_rank)))
        model.module.load_state_dict(snapshot)
        validate(val_loader, model, criterion, local_rank, args)
        return

    if load_names != None and save_names != None:
        # Continue from the checkpoint
        checkpoint = torch.load(os.path.join("./checkpoints", load_names + ".pth.tar"),
                                map_location=torch.device('cuda:{}'.format(local_rank)))
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']
        acc1 = validate(val_loader, model, criterion, local_rank, args)

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        scheduler.step()
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        t2 = time.time()
        print('Time elapsed: ', t2 - t1)
        print('Best top-1 Acc: ', best_acc1)
        if is_best and save_names != None:
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), os.path.join("./snapshots", save_names + ".pth.tar"))

        if args.local_rank == 0:
            print(model.module)
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                }, False, os.path.join("./checkpoints", save_names + ".pth.tar"))

def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        reset_net(model)
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        mean_out = torch.mean(output, dim=1)
        loss = criterion(mean_out, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            reset_net(model)
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images)
            #output = output[:,1:,::]
            mean_out = torch.mean(output, dim=1)
            loss = criterion(mean_out, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()