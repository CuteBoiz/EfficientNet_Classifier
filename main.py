'''
[Train] - [Export excel] - [Export onnx] for Alcon SPC classifier.

author: phatnt
date modify: 2022-01-07
'''
import cv2
import argparse
import os
import csv
import glob 
import numpy as np

from Efnet.model import EfficientNet
from utils.custom_dataset import CustomDataset, Normalize, ToTensor, Resize
from utils.train_utilities import  get_class_weight, train_eff
from utils.report_utilities import create_report, softmax

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def prepare_arguments():
	"""
	Prepare Arguments
	[Train]: Train Efficient Net.

	[Export ONNX]: Export onnx model from trained model.

	[Export Excel Report]: Export report.

	[Infer]: Model inference.
	"""
	parser = argparse.ArgumentParser()
	subparser = parser.add_subparsers(dest="mode")

	train_parser = subparser.add_parser("train")
	
	train_parser.add_argument('--data_path', type=str, required=True, help="Path to train,test,valid text_file's folder")
	train_parser.add_argument('--image_size', type=int, required=True, help='Training image-size')
	train_parser.add_argument('--arch', type=int, default=3, help='EfficientNet architechture(0->8)')
	train_parser.add_argument('--resume', type=str, default=None, help='Path to resume weight for resume previous-training')
	train_parser.add_argument('--max_epoch', type=int, default=20, help='Max epoch number')
	train_parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')

	train_parser.add_argument('--lr', type=float, default=0.001, help='Initial optimizer learing rate')
	train_parser.add_argument('--momentum', type=float, default=0.9, help='Momentum') 
	train_parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay')
	train_parser.add_argument('--gpu', type=int, default=0, help='GPU num')
	train_parser.add_argument('--workers', type=int, default=4, help='Number of data-loading-thread')

	export_onnx_parser = subparser.add_parser("export_onnx")
	export_onnx_parser.add_argument("--weight", type=str, required=True)
	export_onnx_parser.add_argument("--max_batch_size", type=int, default=1)
	export_onnx_parser.add_argument("--export_name", type=str, default=None)
	export_onnx_parser.add_argument("--opset", type=int, default=9)
	export_onnx_parser.add_argument("--verbose", action='store_true', default=False)

	export_excel_parser = subparser.add_parser("export_excel")
	export_excel_parser.add_argument('--data_path', type=str, required=True)
	export_excel_parser.add_argument('--weight', type=str, required=True)
	export_excel_parser.add_argument('--gpu', type=int, default=0)
	export_excel_parser.add_argument('--workers', type=int, default=4)
	export_excel_parser.add_argument('--batch_size', type=int, default=32)

	infer_parser = subparser.add_parser("infer")
	infer_parser.add_argument("--weight", type=str, required=True)
	infer_parser.add_argument("--data", type=str, required=True)
	infer_parser.add_argument("--batch_size", type=int, default=1)
	infer_parser.add_argument("--softmax", action='store_true', default=False)

	return parser.parse_args()
	
def train(args):
	'''
		Train efficient net.
	'''
	print("Train EffcientNet Model")
	assert args.image_size > 0, '[ERROR] Image size must > 0'
	assert args.batch_size > 0, '[ERROR] Batch size must > 0'
	assert args.max_epoch > 0, '[ERROR] Max epoch must > 0'
	assert args.arch >= 0 and args.arch <= 8, '[ERROR] Invalid EfficientNet Architecture (0 -> 8)'
	assert os.path.isdir(args.data_path), '[ERROR] Could not found {}. Or not a directory!'.format(args.data_path)
	if args.resume:
		assert os.path.isfile(args.resume), '[ERROR] Could not found {}.'.format(args.resume)

	#Create Log and Weight save folder
	saved_folder_name = os.path.basename(os.path.dirname(args.data_path))
	log_dir = os.path.join('./logs/', saved_folder_name)
	weight_save_dir = os.path.join('./result/', saved_folder_name)
	print('[INFO] Log Dirs:{}'.format(log_dir))
	print('[INFO] Weights Dirs:{}'.format(weight_save_dir))
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(weight_save_dir, exist_ok=True)
	writer = SummaryWriter(log_dir=log_dir)

	#Load Data 
	train_file = os.path.join(args.data_path, "train.txt")
	valid_file = os.path.join(args.data_path, "valid.txt")
	label_file = os.path.join(args.data_path, "label.txt")
	assert os.path.isfile(train_file), '[ERROR] Could not found train.txt in {}'.format(args.data_path)
	assert os.path.isfile(valid_file), '[ERROR] Could not found valid.txt in {}'.format(args.data_path)
	assert os.path.isfile(label_file), '[ERROR] Could not found label.txt in {}'.format(args.data_path)
	train_classes = []
	with open(label_file, 'r', encoding='utf-8') as label_f:
		lines = label_f.readlines()
	for line in lines:
		line = line.strip()
		train_classes.append(line)
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
	cls_weights = get_class_weight(train_file, len(train_classes))
	print("[INFO] Class Weights:", cls_weights)
	cls_weights = torch.FloatTensor(cls_weights)
	
	#Load Model
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")
	print('[INFO] Model info:')
	print(f'\t + Using Efficientnet-b{args.arch}.')
	print(f'\t + Input image size: {args.image_size} x {args.image_size} x 3.')
	print(f'\t + Output Classes: {len(train_classes)}.')
	model = EfficientNet.from_pretrained('efficientnet-b{}'.format(args.arch), num_classes=len(train_classes), image_size=args.image_size)
	last_epoch = 1

	#Load checkpoints
	if args.resume:
		print("Loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		last_epoch = checkpoint['epoch']
		print("[INFO] Loaded checkpoint '{}' (at epoch {})".format(args.resume, last_epoch))
	model.to(device=device, dtype=torch.float)
	torch.backends.cudnn.benchmark = True

	#Loss & Optimizer
	criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).to(device=device, dtype=torch.float)
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	train_eff(model, train_loader, val_loader, 
				optimizer, criterion, scheduler, args.lr, 
				args.max_epoch, last_epoch, device, writer, 
				args.arch, args.image_size, len(train_classes), weight_save_dir)
	
def export_onnx(args):
	'''
		Export pytorch weight to onnx engine.
	'''
	print("[INFO] Exporting {} to Onnx engine".format(args.weight))
	assert args.max_batch_size > 0
	assert os.path.isfile(args.weight) 

	checkpoint = torch.load(args.weight, map_location='cpu')
	image_size = checkpoint['image_size']
	model = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=image_size)
	
	print('[INFO] Model info:')
	print(f'\t + Architecture:  EfficientNet-b{checkpoint["arch"]}')
	print(f'\t + Input shape: {args.max_batch_size} x 3 x {image_size} x {image_size}')
	print(f'\t + Ouput shape: {checkpoint["nrof_classes"]}')

	model.load_state_dict(checkpoint['state_dict'])
	model.set_swish(memory_efficient=False)
	model.eval()
	dummy_input = torch.randn(args.max_batch_size, 3, image_size, image_size, requires_grad=True)
	saved_name = args.export_name if args.export_name is not None else args.weight.replace('.pth', '.onnx')
	torch.onnx.export(model, dummy_input, saved_name,
					opset_version = args.opset,
					verbose=args.verbose, export_params=True, do_constant_folding=True)
	
	print('{} created!, Exporting Done!'.format(saved_name))

def export_excel(args):
	'''
		Evaluating model via test set then save result into excel file.
	'''
	print("Export Excel Report")
	
	label_file =  os.path.join(args.data_path, "label.txt")
	test_file = os.path.join(args.data_path, "test.txt")
	thres_file = os.path.join(args.data_path, "classifier_thresholds.csv")
	assert os.path.isfile(label_file), '[ERROR] Could not found label.txt in {}'.format(args.data_path)
	assert os.path.isfile(test_file), '[ERROR] Could not found test.txt in {}'.format(args.data_path)
	
	#Get Classes name
	thresholds = []
	train_classes = []
	with open(label_file, 'r') as label_f:
		lines = label_f.readlines()
	for line in lines:
		class_name = line.strip()
		train_classes.append(class_name)
	
	#Create Theshold file if not exist
	if not os.path.isfile(thres_file):
		csv_file = open(thres_file, 'w', encoding='UTF8')
		csv_writer = csv.writer(csv_file)
		for class_name in train_classes:
			csv_writer.writerow(['{}'.format(class_name),'0.2'])
		csv_file.close()
		print('[INFO]: {} created!'.format(thres_file))

	#Get thresholds
	print('[INFO] Train classes / Threshold:')
	with open(thres_file, 'r') as thres_f:
		csv_reader = csv.reader(thres_f, delimiter = ',')
		for row in csv_reader:
			print(row)
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
	print('[INFO] Evaluating model ...')
	create_report(data_folder=args.data_path, evaluate_dict=evaluate_dict, report_save_dir=os.path.dirname(args.weight))

def infer(args):
	'''
		EfficientNet model infer with data(image/video/images folder)
	'''
	assert os.path.isfile(args.weight), "[ERROR] Could not found {}".format(args.weight)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	checkpoint = torch.load(args.weight, map_location='cpu')
	image_size = checkpoint['image_size']
	model = EfficientNet.from_name('efficientnet-b{}'.format(checkpoint['arch']), num_classes=checkpoint['nrof_classes'], image_size=image_size)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device=device, dtype=torch.float)
	model.set_swish(memory_efficient=False)
	model.eval()

	images = []
	if os.path.isfile(args.data):
		extentions = args.data.split('.')[-1]
		if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
			print(args.data)
			image = cv2.imread(args.data)
			images.append(image)
		elif extentions in ['mp4', 'mov', 'wmv', 'mkv', 'avi', 'flv']:
			cap = cv2.VideoCapture(args.data)
			if (cap.isOpened()== False):
  				assert Exception("[ERROR] Error opening video stream or file")
			while(cap.isOpened()):
				ret, frame = cap.read()
				if ret == True:
					images.append(frame)
				else: 
					break
	elif os.path.isdir(args.data):
		files = sorted(glob.glob(os.path.join(args.data, '*')))
		for file in files:
			extentions = file.split('.')[-1]
			if extentions in ['jpg', 'png', 'bmp', 'jpeg']:
				print(file)
				image = cv2.imread(file)
				images.append(image)
			else:
				continue
	else:
		raise Exception(f"[ERROR] Could not load data from {args.data}")

	with torch.no_grad():
		for image in images:
			image = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_AREA)
			image = np.float32(image)
			image = image*(1/255)
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			image = (image - mean) / std
			image = image.transpose((2, 0, 1))
			image = np.asarray([image]).astype(np.float32)
			image = torch.from_numpy(image).to(device=device, dtype=torch.float)
			
			prediction = np.squeeze(model(image).cpu().numpy())
			if args.softmax:
				prediction = softmax(prediction)
			print(prediction)

if __name__ == '__main__':
	
	args = prepare_arguments()

	if args.mode == "train":
		train(args)
	elif args.mode == "export_onnx":
		export_onnx(args)
	elif args.mode == "export_excel":
		export_excel(args)
	elif args.mode == "infer":
		infer(args)
	else:
		raise Exception("Invalid mode. [train] [split] [export_onnx] [export_excel] [infer]")
