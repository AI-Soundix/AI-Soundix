import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from util import *
from dataset import *

import matplotlib.pyplot as plt

import time

# parser
parser = argparse.ArgumentParser(description="ResNet for Audio Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", default="ResNet", type=str, dest="model")

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=32, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./dataset", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./resnet/checkpoint", type=str, dest="ckpt_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")

args = parser.parse_args()

# setting parameters
model = args.model
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir

mode = args.mode

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

print("\n\n\nResNet for Audio Classification")
print("model: %s" % model)
print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("mode: %s" % mode)
print("\n\n\n")

# dataset
if mode == 'train':
    transform = transforms.Compose([transforms.Resize((640,640)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'valid'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([transforms.Resize((640,640)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

# network
if model == 'ResNet':
    net = models.resnet50().to(device)
else:
    net = models.resnet50(weights='DEFAULT').to(device)

# modify last layer for classification
num_class = 6
num_in_features = 2048
net.fc = torch.nn.Linear(num_in_features, num_class).to(device)

# loss funcion
fn_loss = nn.CrossEntropyLoss().to(device)

# optimizer function
optim = torch.optim.Adam(net.parameters(), lr=lr)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_tonumpy_label = lambda x: x.to('cpu').detach().numpy()
fn_denorm = lambda x, mean, std: (x * std) + mean

# train
st_epoch = 0
if mode == 'train':
    start = time.time()

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []
        acc_arr = []

        for batch, (images, labels) in enumerate(loader_train, 1):
            # forward pass
            input = images.to(device)
            label = labels.to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()
            
            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            label = fn_tonumpy_label(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy_label(output)

            output = np.argmax(output, axis=1)
            
            #accuracy 계산
            accuracy = np.mean(np.equal(output, label))
            
            # 손실함수 계산
            loss_arr += [loss.item()]
            acc_arr += [accuracy]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr), np.mean(acc_arr)))

        with torch.no_grad():
            net.eval()
            loss_arr = []
            acc_arr = []

            for batch, (images, labels) in enumerate(loader_val, 1):
                # forward pass
                input = images.to(device)
                label = labels.to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)
                
                label = fn_tonumpy_label(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy_label(output)

                output = np.argmax(output, axis=1)
                
                #accuracy 계산
                accuracy = np.mean(np.equal(output, label))
                
                # 손실함수 계산
                loss_arr += [loss.item()]
                acc_arr += [accuracy]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr), np.mean(acc_arr)))

        if epoch % 5 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
        
    print(f"train의 수행 시간:  {time.time()-start:.4f} sec")

# test
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        acc_arr = []

        for batch, (images, labels) in enumerate(loader_test, 1):
            # forward pass
            input = images.to(device)
            label = labels.to(device)

            start = time.time()
            
            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)
            
            label = fn_tonumpy_label(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy_label(output)

            output = np.argmax(output, axis=1)
            
            #accuracy 계산
            accuracy = np.mean(np.equal(output, label))

            loss_arr += [loss.item()]
            acc_arr += [accuracy]

            print("TEST: BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr), np.mean(acc_arr)))
            
            print(f"batch의 수행 시간:  {time.time()-start:.4f} sec")
                
    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f | ACCURACY %.4f" %
          (batch, num_batch_test, np.mean(loss_arr), np.mean(acc_arr)))