import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
import math
import copy
from tqdm import tqdm


from functions import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--OOD', default="CUB", type=str, help='dataset')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=19930815, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay 5e-4')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--alg', default='RW', type=str, help='name of algorithm')
parser.add_argument('--save', default=False, action='store_true', help='Save model')

parser.add_argument('--C', default=0.001, type=float, help='0.01 0.001 0.0001')
parser.add_argument('--N', default=100, type=int)
args = parser.parse_args()

def train(dataloader, net, classifiers, criterion, optimizer, epoch):
	backbone_fixed = copy.deepcopy(net.backbone)
	ls = cal_loss_dataset(backbone_fixed, classifiers, dataloader, criterion)
	correct = 0
	total = 0
	train_loss = 0.0
	train_acc = 0.0
	net.train()
	for idx, (inputs, targets) in enumerate(dataloader):
		inputs, targets = inputs.cuda(), targets.cuda()
		lb = cal_loss_batch(backbone_fixed, classifiers, inputs, targets, criterion)
		diff = cal_distance(ls, lb)
		outputs_batch_c = cal_weight(diff, args.C)
		outputs = net(inputs)
		loss = criterion(outputs, targets) * outputs_batch_c
		_, predicted = torch.max(outputs.data, 1)
		correct += (predicted == targets).sum().item()
		total += targets.size(0)
		train_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	train_acc = 100.*correct/total
	train_loss = train_loss/idx
	return train_loss, train_acc

def test(testloader, net, criterion):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	test_loss = test_loss/idx
	test_acc = 100.*correct/total
	return test_loss, test_acc

def main():

	trainloader, testloader = load_data(args.dataset, args.dataset, args.batch_size)

	backbone = build_backbone(args.model)
	classifier = build_classifier(backbone.output_size, args.num_classes)
	classifiers = build_classifier_list(backbone.output_size, args.num_classes, args.N)
	net = renet(backbone, classifier, backbone.output_size + 1)
	net.cuda()
	cudnn.benchmark = True

	criterion =  nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

	for epoch in tqdm(range(0, 1)):
		train_loss, train_acc = train(trainloader, net, classifiers, criterion, optimizer, epoch)
	save_model('pth/' + args.name + '.pth', net)

	# ID classification
	test_loss, test_acc = test(testloader, net, criterion)
	print("ID classification performance：", test_acc)

	# OOD detection
	testloader_OOD = load_data(args.dataset, args.OOD, 100)
	ID_scores = msp_detector(net, testloader)
	OOD_scores = msp_detector(net, testloader_OOD)
	test_auroc = auroc(ID_scores, OOD_scores, 100000)*100
	print("OOD detection performance：", test_auroc)

if __name__ == '__main__':
	main()