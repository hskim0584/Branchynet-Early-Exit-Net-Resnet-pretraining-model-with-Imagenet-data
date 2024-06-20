import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', '--d', dest='data', metavar='DIR', default='./data', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet101)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=1, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec11 = 0.0
best_prec21 = 0.0
best_prec31 = 0.0


def main():
    global args, best_prec11, best_prec21, best_prec31
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use cuda
    # model.cuda()
    # model = nn.DataParallel(model).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec21 = checkpoint['best_prec21']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_loader, val_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec11, prec21, prec31 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        is_best = prec11 + prec21 + prec31 > best_prec11 + best_prec21 + best_prec31
        if (is_best):
            best_prec11 = prec11
            best_prec21 = prec21
            best_prec31 = prec31

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec21': best_prec21,
            'optimizer': optimizer.state_dict()
        }, is_best, args.arch + '.pth')
    # torch.save(model.module.conv1.state_dict(), './conv1.pth')
    # torch.save(model.module.bn1.state_dict(), './bn1.pth')
    # torch.save(model.module.relu.state_dict(), './relu.pth')
    # torch.save(model.module.maxpool.state_dict(), './maxpool.pth')
    # torch.save(model.module.layer1.state_dict(), './layer1.pth')
    # torch.save(model.module.layer2.state_dict(), './layer2.pth')
    # torch.save(model.module.layer3.state_dict(), './layer3.pth')
    # torch.save(model.module.InterFE1.state_dict(), './InterFE1.pth')
    # torch.save(model.module.InterFE2.state_dict(), './InterFE2.pth')
    # torch.save(model.module.layer4.state_dict(), './layer4.pth')
    # torch.save(model.module.state_dict(), './FinalModel.pth')


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    top11 = AverageMeter()
    top15 = AverageMeter()

    top21 = AverageMeter()
    top25 = AverageMeter()

    top31 = AverageMeter()
    top35 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    count = 0
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)

        # compute output
        output1, output2, output3 = model(input)
        # print("output1", output1)
        # print("output2", output2)
        # print("output3", output3)
        loss1 = criterion(output1, target)
        loss2 = criterion(output2, target)
        loss3 = criterion(output3, target)

        # measure accuracy and record loss
        prec11, prec15 = accuracy(output1.data, target, topk=(1, 2))
        prec21, prec25 = accuracy(output2.data, target, topk=(1, 2))
        prec31, prec35 = accuracy(output2.data, target, topk=(1, 2))

        losses1.update(loss1.item(), input.size(0))
        losses2.update(loss2.item(), input.size(0))
        losses3.update(loss3.item(), input.size(0))

        top11.update(prec11[0], input.size(0))

        top21.update(prec21[0], input.size(0))

        top31.update(prec31[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'

                  'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                  'Prec1@1 {top11.val:.3f} ({top11.avg:.3f})\n'

                  'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                  'Prec2@1 {top21.val:.3f} ({top21.avg:.3f})\n'

                  'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                  'Prec3@1 {top31.val:.3f} ({top31.avg:.3f})\n'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time,
                loss1=losses1, top11=top11,
                loss2=losses2, top21=top21,
                loss3=losses3, top31=top31))


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()

    losses1 = AverageMeter()
    top11 = AverageMeter()

    losses2 = AverageMeter()
    top21 = AverageMeter()

    losses3 = AverageMeter()
    top31 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output1, output2, output3 = model(input)

            loss1 = criterion(output1, target)
            loss2 = criterion(output2, target)
            loss3 = criterion(output3, target)

            # measure accuracy and record loss
            prec11, _ = accuracy(output1.data, target, topk=(1, 2))
            losses1.update(loss1.item(), input.size(0))
            top11.update(prec11[0], input.size(0))

            prec21, _ = accuracy(output2.data, target, topk=(1, 2))
            losses2.update(loss2.item(), input.size(0))
            top21.update(prec21[0], input.size(0))

            prec31, _ = accuracy(output3.data, target, topk=(1, 2))
            losses3.update(loss3.item(), input.size(0))
            top31.update(prec31[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'

                      'Loss1 {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'Prec@11 {top11.val:.3f} ({top11.avg:.3f})\n'

                      'Loss2 {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Prec@21 {top21.val:.3f} ({top21.avg:.3f})\n'

                      'Loss3 {loss3.val:.4f} ({loss3.avg:.4f})\t'
                      'Prec@31 {top31.val:.3f} ({top31.avg:.3f})\n'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss1=losses1, top11=top11,
                    loss2=losses2, top21=top21,
                    loss3=losses3, top31=top31))

    print(' * Prec@11 {top11.avg:.3f}'.format(top11=top11))
    print(' * Prec@21 {top21.avg:.3f}'.format(top21=top21))
    print(' * Prec@31 {top31.avg:.3f}'.format(top31=top31))

    return top11.avg, top21.avg, top31.avg


if __name__ == '__main__':
    main()
