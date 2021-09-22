'''
[Train] - [Export excel] - [Export onnx] for Alcon SPC classifier.

author: phatnt
date modify: 2021-09-20
'''

import argparse
import os
import time
import csv
import numpy as np
from datetime import datetime
from Efnet.model import EfficientNet
from utils.custom_dataset import CustomDataset, Normalize, ToTensor, Resize
from utils.utilities import evaluate, get_class_weight, adjust_learning_rate, save_checkpoint, create_report

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def prepare_arguments(parser):
	"""
	Prepare Arguments
	[Train]: Train Efficient Net.

	[Export ONNX]: Export onnx model from trained model.

	[Export Excel Report]: Export report.
	"""
	subparser = parser.add_subparsers(dest="mode")

	train_parser = subparser.add_parser("train")
	train_parser.add_argument('--gpu', type=int, required=True, help='GPU id')
	train_parser.add_argument('--path', type=str, required=True, help="Path to train,test,valid text_file's folder")
	train_parser.add_argument('--workers', type=int, default=4, help='Number of data-loading-thread')
	train_parser.add_argument('--image_size', type=int, required=True, help='Training image-size')
	train_parser.add_argument('--arch', type=int, default=3, help='EfficientNet architechture(0-7)')
	train_parser.add_argument('--resume', type=str, default='', help='Path to resume weight for resume previous-training')
	train_parser.add_argument('--epoch', type=int, default=20, help='Max epoch number')
	train_parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')

	train_parser.add_argument('--lr', type=float, default=0.001, help='Optimizer Learing rate')
	train_parser.add_argument('--momentum', type=float, default=0.9) 
	train_parser.add_argument('--weight-decay', default=1e-4, type=float)

	export_onnx_parser = subparser.add_parser("export_onnx")
	export_onnx_parser.add_argument("--weight", type=str, required=True)
	export_onnx_parser.add_argument("--max_batch_size", type=int, default=1)
	export_onnx_parser.add_argument("--export_name", type=str, required=True)

	export_excel_parser = subparser.add_parser("export_excel")
	export_excel_parser.add_argument('--gpu', type=int, required=True)
	export_excel_parser.add_argument('--path', type=str, required=True)
	export_excel_parser.add_argument('--weight', type=str, required=True)
	export_excel_parser.add_argument('--init_thres', action='store_true', default=False)
	export_excel_parser.add_argument('--workers', type=int, default=4)
	export_excel_parser.add_argument('--batch_size', type=int, default=32)
	
def train(args):
	print("Train EffcientNet Model")
	assert args.image_size > 0, '[ERROR] Image size must > 0'
	assert args.batch_size > 0, '[ERROR] Batch size must > 0'
	assert args.arch >= 0 and args.arch <= 7
	if args.resume:
		assert os.path.isfile(args.resume), '[ERROR] Could not found {}.'.format(args.resume)

	log_dir = './logs/{}/'.format(datetime.today().strftime('%Y-%m-%d'))
	result_dir = './result/{}/'.format(datetime.today().strftime('%Y-%m-%d'))
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(result_dir, exist_ok=True)

	#Load Data 
	train_file = os.path.join(args.path, "train.txt")
	valid_file = os.path.join(args.path, "train.txt")
	label_file = os.path.join(args.path, "label.csv")
	assert os.path.isfile(train_file), '[ERROR] Could not found train.txt in {}'.format(args.path)
	assert os.path.isfile(valid_file), '[ERROR] Could not found valid.txt in {}'.format(args.path)
	assert os.path.isfile(label_file), '[ERROR] Could not found label.csv in {}'.format(args.path)

	train_classes = []
	with open(label_file, 'r', encoding='utf-8') as label_f:
		csv_reader = csv.reader(label_f, delimiter = ',')
		for row in csv_reader:
			train_classes.append(row[0])
	print('[INFO] Number of classes:', len(train_classes))
	print('[INFO] Training class: ', train_classes)

	train_transforms = transforms.Compose([Resize(args.image_size), Normalize(), ToTensor()])
	train_loader = torch.utils.data.DataLoader(CustomDataset(train_file, train_transforms),
												batch_size=args.batch_size, shuffle=True,
												num_workers=args.workers, pin_memory=True, sampler=None)
	val_transforms = transforms.Compose([Resize(args.image_size), Normalize(), ToTensor()])
	val_loader = torch.utils.data.DataLoader(CustomDataset(valid_file, val_transforms),
												batch_size=args.batch_size, shuffle=False,
												num_workers=args.workers, pin_memory=True)
	
	#Load Model
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")
	print('[INFO] Using efficientnet-b{}'.format(args.arch))
	model = EfficientNet.from_pretrained('efficientnet-b{}'.format(args.arch), num_classes=len(train_classes), image_size=args.image_size)
	last_epoch = 0

	if args.resume:
		print("Loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		print("[INFO] Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))
		last_epoch = checkpoint['epoch'] - 1
	model.to(device=device, dtype=torch.float)
	torch.backends.cudnn.benchmark = True

	cls_weights = get_class_weight(train_file, len(train_classes))
	print("Class Weight:", cls_weights)
	cls_weights = torch.FloatTensor(cls_weights)
	criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).to(device=device, dtype=torch.float)
	
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	writer = SummaryWriter(log_dir=log_dir)
	best_loss = 1e5
	best_acc = 0

	for epoch in range(args.epoch):
		if epoch < last_epoch:
			continue
		start = time.time()
		model.train()
		epoch_loss = []
		is_best_loss = False
		is_best_acc = False
		adjust_learning_rate(optimizer, epoch, args)

		for i, sample_batched in enumerate(train_loader):
			images = sample_batched['image']
			target = sample_batched['target']
			if args.gpu is not None:
				images = images.to(device=device, dtype=torch.float)
			target = target.to(device=device, dtype=torch.float)
			output = model(images)
			loss = criterion(output, target.long())
			if loss == 0 or not torch.isfinite(loss):
				continue
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			epoch_loss.append(float(loss))

			if i % 100 == 0:
				print("Epoch {}/{}: Step{}, Loss: {}".format(epoch+1, args.epoch, i, loss))
			if i % 10000 == 0:
				writer.add_scalar('Training Loss', loss, epoch * len(train_loader) + i)

		scheduler.step(np.mean(epoch_loss))
		end = time.time()
		accuracy = evaluate(val_loader, model, device)
		current_lr = optimizer.param_groups[0]['lr']
		writer.add_scalar('Training Learning rate', current_lr, epoch+1)
		writer.add_scalar('Training Accuracy', accuracy, epoch+1)

		print("Time: {}, Epoch {}/{}: Loss: {}, Accuracy:{}%".format(end-start, epoch+1, args.epoch, loss, accuracy))
		if (loss < best_loss):
			best_loss = loss
			is_best_loss = True
		if (accuracy > best_acc):
			best_acc = accuracy
			is_best_acc = True
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'image_size': args.image_size,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'nrof_classes': len(train_classes)},
			saved_path=result_dir, is_best_loss=is_best_loss, is_best_acc=is_best_acc)
		is_best_loss = False
		is_best_acc = False
	writer.close()
	
	
def export_onnx(args):
	print("Export Onnx Model")
	assert args.max_batch_size > 0
	assert os.path.isfile(args.weight) 

	checkpoint = torch.load(args.weight, map_location='cpu')
	image_size = checkpoint['image_size']
	model = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=image_size)
	
	model.load_state_dict(checkpoint['state_dict'])
	model.set_swish(memory_efficient=False)
	model.eval()
	dummy_input = torch.randn(args.max_batch_size, 3, image_size, image_size, requires_grad=True)
	model_out = model(dummy_input)
	torch.onnx.export(model, dummy_input, args.name, verbose=True, export_params=True, do_constant_folding=True)

def export_excel(args):
	print("Export Excel Report")

	label_file =  os.path.join(args.path, "label.txt")
	test_file = os.path.join(args.path, "test.txt")
	thres_file = os.path.join(args.path, "classifier_thresholds.csv")
	assert os.path.isfile(label_file), '[ERROR] Could not found label.txt in {}'.format(args.path)
	assert os.path.isfile(test_file), '[ERROR] Could not found test.txt in {}'.format(args.path)
	
	#Get Classes name
	thresholds = []
	train_classes = []
	with open(label_file, 'r', encoding='utf-8') as label_f:
		lines = label_f.readlines()
	for line in lines:
		class_name = line.strip()
		train_classes.append(class_name)
	
	#Create Theshold file if not exist
	if not args.init_thres:
		assert os.path.isfile(thres_file), '[ERROR] Could not found threshold.txt in {}. \n Add --init_thres to create thresholds.csv'.format(args.path)
	else:
		csv_file = open(thres_file, 'w', encoding='UTF8')
		csv_writer = csv.writer(csv_file)
		for class_name in train_classes:
			csv_writer.writerow(['{}'.format(class_name),'0.2'])
		csv_file.close()
		print('[INFO]: {} created!'.format(thres_file))

	#Get thresholds 
	with open(thres_file, 'r') as thres_f:
		csv_reader = csv.reader(thres_f, delimiter = ',')
		for row in csv_reader:
			threshold = float(row[1])
			thresholds.append(threshold)
	assert len(thresholds) == len(train_classes)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(args.gpu)
	checkpoint = torch.load(args.weight, map_location=device)
	image_size = checkpoint['image_size']
	model = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=image_size)

	data_transforms = transforms.Compose([Resize(image_size), Normalize(), ToTensor()])
	data_loader = torch.utils.data.DataLoader(CustomDataset(test_file, data_transforms),
												batch_size=args.batch_size, shuffle=False,
												num_workers=args.workers, pin_memory=True)

	model.load_state_dict(checkpoint['state_dict'])
	model.to(device=device, dtype=torch.float)
	model.set_swish(memory_efficient=False)
	model.eval()
	evaluate_dict ={'model': model,
					'data_loader': data_loader,
					'train_classes': train_classes,
					'thresholds': thresholds,
					'device': device}

	create_report(data_folder=args.path, evaluate_dict=evaluate_dict, report_save_dir=os.path.dirname(args.weight))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	prepare_arguments(parser)
	args = parser.parse_args()

	if args.mode == "train":
		train(args)
	elif args.mode == "export_onnx":
		export_onnx(args)
	elif args.mode == "export_excel":
		export_excel(args)
	else:
		raise Exception("Invalid mode. [train] [split] [export_onnx] [export_excel]")
