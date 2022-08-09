import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from PIL import Image
import os
import sys
import os.path
import numpy as np
import math
import random
from numpy.random import randint
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet18
from models.senet import SENet18
from models.shufflenetv2 import ShuffleNetV2

class LoadDataset(Dataset):
	"""docstring for LoadDataset"""
	def __init__(self,  root, list_file='train', transform=None, target_transform=None, full_dir=True):
		super(LoadDataset, self).__init__()
		self.root = root
		self.list_file = list_file
		self.transform = transform
		self.target_transform = target_transform
		self.full_dir = full_dir
		self._parse_list()

	def _load_image(self, directory):
		if self.full_dir:
			return Image.open(directory).convert('RGB')
		else:
			return Image.open(os.path.join(self.root, 'data', directory)).convert('RGB')

	def _parse_list(self):
		self.data_list = [LoadRecord(x.strip().split(' ')) for x in open(os.path.join(self.root, self.list_file))]

	def __getitem__(self, index):
		record = self.data_list[index]

		return self.get(record)

	def get(self, record, indices=None):
		img = self._load_image(record.path)

		process_data = self.transform(img)
		if not self.target_transform == None:
			process_label = self.target_transform(record.label)
		else:
			process_label = record.label

		return process_data, process_label

	def __len__(self):
		return len(self.data_list)

class LoadRecord(object):
	"""docstring for LoadRecord"""
	def __init__(self, data):
		super(LoadRecord, self).__init__()
		self._data = data

	@property
	def path(self):
		return self._data[0]

	@property
	def label(self):
		return int(self._data[1])

def getCIFAR10(transform, batch_size=128, shuffle=True, train=True, test=True):

	ds = []

	if train:
		trainset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		testset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getCIFAR100(transform, batch_size=128, shuffle=True, train=True, test=True):

	ds = []

	if train:
		trainset = datasets.CIFAR100(root='data/CIFAR100', train=True, download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		testset = datasets.CIFAR100(root='data/CIFAR100', train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def getSVHN(transform, batch_size=128, shuffle=True, train=True, test=True):

	ds = []

	if train:
		trainset = datasets.SVHN(root='data', split='train', download=True, transform=transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(trainloader)
	if test:
		testset = datasets.SVHN(root='data', split='test', download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
		ds.append(testloader)

	ds = ds[0] if len(ds) == 1 else ds
	return ds

def get_transform(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'CIFAR100':
		mean = (0.5071, 0.4865, 0.4409)
		std = (0.2673, 0.2564, 0.2762)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'SVHN':
		mean = (0.4377, 0.4438, 0.4728)
		std = (0.1980, 0.2010, 0.1970)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	return transform_train, transform_test

def load_train_ID(transform_train, transform_test, dataset, batch_size):
	if dataset == 'CIFAR10':
		trainloader = getCIFAR10(transform_train, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getCIFAR10(transform_test, batch_size=100, shuffle=False, train=False, test=True)
	elif dataset == 'CIFAR100':
		trainloader = getCIFAR100(transform_train, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getCIFAR100(transform_test, batch_size=100, shuffle=False, train=False, test=True)
	elif dataset == 'SVHN':
		trainloader = getSVHN(transform_train, batch_size=batch_size, shuffle=True, train=True, test=False)
		testloader = getSVHN(transform_test, batch_size=100, shuffle=False, train=False, test=True)
	return trainloader, testloader

def load_data(dataset_ID, dataset, batch_size):
	transform_train, transform_test = get_transform(dataset_ID)
	if dataset_ID == dataset:
		trainloader, testloader = load_train_ID(transform_train, transform_test, dataset, batch_size)
		return trainloader, testloader
	else:
		testloader = getAllOOD(root = '../data/' + dataset + '/',transform = transform_test, 
			batch_size=100, shuffle=False, glist=False)
		return testloader

def build_backbone(model):
	if model == 'resnet':
		backbone = ResNet18()
	elif model == 'senet':
		backbone = SENet18()
	elif model == 'shufflenet':
		backbone = ShuffleNetV2()
	return backbone

class LinearClassifier(nn.Module):
	"""docstring for LinearClassifier"""
	def __init__(self, output_size, num_classes):
		super(LinearClassifier, self).__init__()
		self.output_size = output_size
		self.num_classes = num_classes
		self.h = nn.Linear(output_size, self.num_classes)
	def forward(self, output_backbone):
		output = self.h(output_backbone)
		return output

def build_classifier(output_size, num_classes=10):
	classifier = LinearClassifier(output_size, num_classes)
	classifier.cuda()
	return classifier

def build_classifier_list(output_size, num_classes, num):
	classifiers = []
	for i in range(num):
		classifiers.append(build_classifier(output_size, num_classes))
	return classifiers

class renet(nn.Module):
	"""docstring for renet"""
	def __init__(self, backbone, classifier, output_size):
		super(renet, self).__init__()
		self.backbone = backbone
		self.classifier = classifier
		self.hidden_state = output_size
		self.fc1 = nn.Linear(output_size, self.hidden_state)
		self.fc2 = nn.Linear(self.hidden_state, 1)
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
	def forward(self, x, z = None):
		output_backbone = self.backbone(x)
		output = self.classifier(output_backbone)
		return output

def cal_distance(B, S):
	diff = []
	for i in range(len(B)):
		diff.append(np.abs(B[i] - S[i]))
	return diff

def cal_weight(diff, C):
	Sigmoid = nn.Sigmoid()
	Tanh = nn.Tanh()
	m = nn.ReLU()
	distance = torch.from_numpy(np.array(max(diff)).astype(np.float32))
	outputs_batch = torch.tensor([Tanh(C / distance).item()]).cuda()
	# outputs_batch = torch.tensor([Sigmoid(args.C / distance).item()]).cuda()
	# outputs_batch = torch.tensor([m(1 - distance * 100).item()]).cuda()
	return outputs_batch

def cal_loss_dataset(backbone, classifiers, dataloader, criterion):
	backbone.eval()
	ls = [0.0 for i in range(len(classifiers))]
	num_sum = 0.0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs_backbone = backbone(inputs)
			num_sum += targets.size(dim = 0)
			for i in range(len(classifiers)):
				classifier = classifiers[i]
				outputs = classifier(outputs_backbone)
				loss = criterion(outputs, targets) * targets.size(dim = 0)
				ls[i] += loss.item()
	for i in range(len(ls)):
		ls[i] = ls[i] / num_sum
	return ls

def cal_loss_batch(backbone, classifiers, inputs, targets, criterion):
	backbone.eval()
	lb = []
	outputs_backbone = backbone(inputs)
	with torch.no_grad():
		for i in range(len(classifiers)):
			classifier = classifiers[i]
			outputs = classifier(outputs_backbone)
			loss = criterion(outputs, targets)
			lb.append(loss.item())
	return lb

def save_model(pth_name, net):
	torch.save(net.state_dict(), pth_name)

def getAllOOD(root, transform, batch_size=128, shuffle=False, glist=False):
	if glist:
		generate_all_list(root=root)
	dataset = LoadDataset(root=root, list_file='all_list.txt', transform=transform, full_dir=False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
	return dataloader

def generate_all_list(root='../data224/CUB_200_2011/'):
	path = root + 'data'
	classname = os.listdir(path)
	classnum = len(classname)
	#random.shuffle(classname)
	for i in range(classnum):
		images = os.listdir(os.path.join(path, classname[i]))
		m = 'w' if i == 0 else 'a'
		with open(os.path.join(root, 'all_list.txt'), m) as f:
			for j in range(len(images)):
				f.write(classname[i] + '/')
				f.write(images[j] + ' ' + str(i))
				f.write('\n')

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def msp_detector(net, dataloader):
	net.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, _) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net(inputs)
			softmax_vals, predicted = torch.max(F.softmax(outputs.data, dim=1), dim=1)
			res = np.append(res, softmax_vals.cpu().numpy())
	return res