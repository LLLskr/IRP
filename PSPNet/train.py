#!/usr/local/bin/python3
import os
import argparse
import numpy as np
from criterion import CrossEntropyLoss2d,Relabel,ToLabel
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from eval import get_mean_acc_and_IoU,get_pixel_acc
from dataset.lip import LIPWithClass,LIP
from net.pspnet import PSPNet
import matplotlib.pyplot as plt

#网络的配置参数，这里使用的backone是densenet
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}
#配置参数
parser = argparse.ArgumentParser(description="Human Parsing")
parser.add_argument('--data-path', type=str,default='train/train',
                    help='Path to dataset folder') #数据地址
parser.add_argument('--val-path', type=str,default='train/val',
                    help='Path to dataset folder') #数据地址
parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')#网络backone
parser.add_argument('--snapshot', type=str, default=None, help='Path to pre-trained weights')#预训练参数地址
parser.add_argument('--batch-size', type=int, default=4, help="Number of images sent to the network in one step.")#反向传播批次
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs to run')#训练批次
parser.add_argument('--crop_x', type=int, default=256, help='Horizontal random crop size')
parser.add_argument('--crop_y', type=int, default=256, help='Vertical random crop size')
parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
parser.add_argument('--start-lr', type=float, default=0.0001, help='Learning rate')#开始训练时的优化器学习率
parser.add_argument('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
args = parser.parse_args()

#初始化网络
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)#数据并行
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))#加载网络参数
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()#将网络放到GPU
    return net, epoch

#数据的转制
def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.Resize((256, 256), 0),
        # transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
        ToLabel(),
        Relabel(255, 1),
    ]

    val_transform_gt_list = [
        transforms.Resize((256, 256), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]

    data_transforms = {
        'img': transforms.Compose(transform_image_list),
        'gt': transforms.Compose(transform_gt_list),
        'val':transforms.Compose(val_transform_gt_list)
    }
    return data_transforms

#加载数据
def get_dataloader():
    data_transform = get_transform()
    train_loader = DataLoader(LIP(root=args.data_path, transform=data_transform['img'],
                                  gt_transform=data_transform['gt']),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                              )
    val_loader = DataLoader(LIP(root=args.val_path,transform=data_transform['img'],
                                gt_transform=data_transform['val']),
                                batch_size=1,
                                shuffle=False)


    return train_loader,val_loader

if __name__ == '__main__':

    models_path = os.path.join('./checkpoints', args.backend)#保存网络参数的路径
    os.makedirs(models_path, exist_ok=True)

    train_loader,val_loader = get_dataloader()

    net, starting_epoch = build_network(args.snapshot, args.backend)
    optimizer = optim.Adam(net.parameters(), lr=args.start_lr) #优化器
    # optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9,
    #                weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in args.milestones.split(',')])


    seg_criterion = CrossEntropyLoss2d(weight=None)#损失函数
    epoch_losses = []
    net.train()


    weight_save_path = "checkpoints/densenet/PSPNet_last" #网络参数地址

    #加载网络参数
    try:
        net.load_state_dict(torch.load(weight_save_path))
        print("加载成功")
    except:
        print('加载失败')
    #开始训练
    Loss_list = []
    Accuracy_list = []
    for epoch in range(1+starting_epoch, 1+starting_epoch+args.epochs):
        net.train()
        for count, (x, y) in enumerate(train_loader):
            # input data
            x,y = x.cuda(),y.cuda()
            # forward
            out, out_cls = net(x)
            seg_loss = seg_criterion(out,y)
            loss = seg_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print
            epoch_losses.append(loss.item())
            status = '[{0}] step = {1}/{2}, loss = {3:0.4f} avg = {4:0.4f}, LR = {5:0.7f}'.format(
                epoch, count, len(train_loader),
                loss.item(), np.mean(epoch_losses), scheduler.get_lr()[0])
            print(status)

        Loss_list.append(np.mean(epoch_losses))

        scheduler.step()
        net.eval()
        with torch.no_grad():
            for index, (img, gt) in enumerate(val_loader):

                pred_seg, pred_cls = net(img.cuda())
                pred_seg = pred_seg[0]
                pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
                pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8).reshape((256, 256, 1))
                gt = np.asarray(gt.numpy(), dtype=np.uint8).transpose(1, 2, 0)

                _mean_acc, _mean_IoU = get_mean_acc_and_IoU(pred, gt, 2)

        Accuracy_list.append(_mean_acc)

        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", 'last']))) #保存网络参数

    x1 = range(0, 100)
    x2 = range(0, 100)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig("accuracy_loss_test.jpg")
    plt.show()



