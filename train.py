import argparse
import os
import shutil
import os
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from efficientnet_pytorch import EfficientNet

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from custom_dataset import CustomDataset, Normalize, ToTensor, Resize

def parser_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--data_path', type=str)
	parser.add_argument('--imgsz', type=int, default=None)
	parser.add_argument('--channels',type=int, default=3)
	parser.add_argument('--arch', type=int, default=3, help='EfficientNet architechture(0->8)')
	parser.add_argument('--resume', type=str, default=None, help='Path to resume weight for resume previous-training')
	parser.add_argument('--epoch', type=int, default=20)
	parser.add_argument('--batch', type=int, default=16)

	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--momentum', type=float, default=0.9) 
	parser.add_argument('--weight-decay', default=1e-4, type=float)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--workers', type=int, default=4)

	return parser.parse_args()

def save_checkpoint(state, saved_path, is_best_loss, is_best_acc):
	'''
		Save training model in a specific folder.
		Args:
			state: torch state dict.
			saved_path: model saved place.
			is_best_loss: is this model reach lowest loss?
			is_best_acc: is this model reach highest accuracy on valid set?
	'''
	assert state is not None, '[ERROR]: state dict is none!'
	os.makedirs(saved_path, exist_ok=True)

	saved_file = os.path.join(saved_path, 'model_last.pth') 
	torch.save(state, saved_file)
	if is_best_loss:
		shutil.copyfile(saved_file, saved_file.replace('last', 'best_loss'))
	if is_best_acc:
		shutil.copyfile(saved_file, saved_file.replace('last', 'best_acc'))
	print('[INFO] Checkpoint saved!')

def get_class_weight(train_file, nrof_classes):
	'''
		Get class weight array. In order to deal with imbalance dataset.
		Args: 
			train_file: txt file with format ("image_path", label_id) each line.
			nrof_classes: number of training classes.
		Return:
			An class weight array (length = nrof_classes)
	'''
	assert os.path.isfile(train_file), '[ERROR]: {} not found!'.format(train_file)
	assert isinstance(nrof_classes, int) and nrof_classes >= 0, '[ERROR]: nrof_classes is not valid!'
	
	# Count quantity of each class
	class_quantities = np.zeros(nrof_classes, dtype=np.int32)
	class_weights = []
	with open(train_file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		target = int(line.split("\"")[2].strip())
		assert target < nrof_classes
		class_quantities[target]+=1

	# Caculate class-weight
	for quantity in class_quantities:
		cls_weight = np.sum(class_quantities)/(nrof_classes*quantity)
		class_weights.append(cls_weight)
	return class_weights

def adjust_learning_rate(optimizer, epoch_num, init_lr):
	'''
		Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
		Args:
			optimizer: training optimizer.
			epoch_num: current epoch num.
			learning_rate: init learning rate
	'''
	lr = init_lr * (0.1 ** (epoch_num // 5))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def evaluate(val_loader, model, device):
	'''
		Caculate Accuracy from validation set.
		Args:
			val_loader: torch dataloader format.
			model: Efficient training model.
			device: current training gpu.
		Return:
			accuracy: Accuracy of model base on validation set.
	'''
	print('[INFO] Evaluating ...')
	model.eval()
	True_Cases = 0
	Cases = 0
	with torch.no_grad():
		for i, sample_batched in enumerate(val_loader):
			images = sample_batched['image']
			targets = sample_batched['target']
			images = images.to(device=device, dtype=torch.float)

			output = model(images)
			output = output.cpu().data.numpy().argmax(axis=1)
			targets = np.array(targets)
			for i in range(len(output)):
				Cases += 1
				if int(output[i]) == int(targets[i]):
					True_Cases += 1
		accuracy = (True_Cases/Cases)*100
	return accuracy

def train(args):
	'''
		Train efficient net.
	'''
	assert args.image_size > 0, '[ERROR] Image size must > 0'
	assert args.batch_size > 0, '[ERROR] Batch size must > 0'
	assert args.max_epoch > 0, '[ERROR] Max epoch must > 0'
	assert args.arch >= 0 and args.arch <= 8, '[ERROR] Invalid EfficientNet Architecture (0 -> 8)'
	assert os.path.isdir(args.data_path), f'[ERROR] Could not found {args.data_path}. Or not a directory!'
		
	# Load model
	if torch.cuda.is_available():
		device = torch.device("cuda")
		torch.cuda.set_device(args.device)
	else:
		device = torch.device("cpu")
	model_name = f'efficientnet-b{args.arch}'
	if args.imgsz is None:
		model = EfficientNet.from_pretrained(model_name, in_channels=args.channels, num_classes=len(train_classes))
		imgsz = model.get_image_size(model_name)
	else:
		imgsz = args.imgsz 
		model = EfficientNet.from_pretrained(model_name, in_channels=args.channels, num_classes=len(train_classes), image_size=imgsz)

	# Load data 
	train_file = os.path.join(args.data_path, "train.txt")
	valid_file = os.path.join(args.data_path, "val.txt")
	label_file = os.path.join(args.data_path, "label.txt")
	assert os.path.isfile(train_file), f'[ERROR] Could not found train.txt in {args.data_path}'
	assert os.path.isfile(valid_file), f'[ERROR] Could not found val.txt in {args.data_path}'
	assert os.path.isfile(label_file), f'[ERROR] Could not found label.txt in {args.data_path}'
	train_classes = []
	with open(label_file, 'r', encoding='utf-8') as label_f:
		lines = label_f.readlines()
	for line in lines:
		train_classes.append(line.strip())
	print('[INFO] Number of classes:', len(train_classes))
	print('[INFO] Training class: ', train_classes)
	train_transforms = transforms.Compose([Resize(imgsz), Normalize(), ToTensor()])
	train_loader = torch.utils.data.DataLoader(CustomDataset(train_file, train_transforms),
												batch_size=args.batch, shuffle=True,
												num_workers=args.workers, pin_memory=True, sampler=None)
	val_transforms = transforms.Compose([Resize(imgsz), Normalize(), ToTensor()])
	val_loader = torch.utils.data.DataLoader(CustomDataset(valid_file, val_transforms),
												batch_size=args.batch, shuffle=True,
												num_workers=args.workers, pin_memory=True)
	cls_weights = get_class_weight(train_file, len(train_classes))
	print("[INFO] Class Weights:", cls_weights)
	cls_weights = torch.FloatTensor(cls_weights)

	print('[INFO] Model info:')
	print(f'\t + Using Efficientnet-b{args.arch}.')
	print(f'\t + Input image size: {imgsz} x {imgsz} x {args.channels}.')
	print(f'\t + Output Classes: {len(train_classes)}.')
	
	last_epoch = 1
	# Resume checkpoints
	if args.resume:
		assert os.path.isfile(args.resume), f'[ERROR] Could not found {args.resume}.'
		print(f"[INFO] Loading checkpoint '{args.resume}'")
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		last_epoch = checkpoint['epoch']
		print(f"[INFO] Loaded checkpoint '{args.resume}' (at epoch {last_epoch})")
	model.to(device=device, dtype=torch.float)
	torch.backends.cudnn.benchmark = True

	# Loss & Optimizer
	criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).to(device=device, dtype=torch.float)
	optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	if args.resume:
		optimizer.load_state_dict(checkpoint['optimizer'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	# Create log and weight saved folder
	saved_folder_name = os.path.basename(os.path.dirname(args.data_path))
	log_dir = os.path.join('./logs/', saved_folder_name)
	weight_save_dir = os.path.join('./result/', saved_folder_name)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(weight_save_dir, exist_ok=True)
	writer = SummaryWriter(log_dir=log_dir)
	print(f'[INFO] Log Dirs: {log_dir} Created!')
	print(f'[INFO] Weights Dirs: {weight_save_dir} Created')
	
	# Train
	init_lr = args.lr
	max_epoch = args.epoch+1
	nrof_classes = len(train_classes)
	best_loss = 1e5
	best_acc = 0
	try:
		for epoch in range(1, max_epoch):
			if epoch < last_epoch:
				continue
			model.train()
			epoch_loss = []
			is_best_loss = False
			is_best_acc = False
			adjust_learning_rate(optimizer, epoch, init_lr)
			progress_bar = tqdm(total=len(train_loader))

			for i, sample_batched in enumerate(train_loader):
				images = sample_batched['image']
				target = sample_batched['target']
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
				progress_bar.set_description( f"Epoch: {epoch}/{max_epoch}. Loss: {loss:.5f}.")
				progress_bar.update(1)

			scheduler.step(np.mean(epoch_loss))
			
			accuracy = evaluate(val_loader, model, device)
			current_lr = optimizer.param_groups[0]['lr']
			writer.add_scalar('Training Loss', loss, epoch * len(train_loader) + i)
			writer.add_scalar('Training Learning rate', current_lr, epoch)
			writer.add_scalar('Training Accuracy', accuracy, epoch)

			print(f"[INFO] Epoch {epoch}: Loss: {loss:.5f}, Accuracy: {accuracy:.2f}%")
			if (loss < best_loss):
				best_loss = loss
				is_best_loss = True
			if (accuracy > best_acc):
				best_acc = accuracy
				is_best_acc = True
			save_checkpoint({
				'epoch': epoch,
				'arch': args.arch,
				'image_size': imgsz,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'nrof_classes': nrof_classes},
				saved_path=weight_save_dir, is_best_loss=is_best_loss, is_best_acc=is_best_acc)
			is_best_loss = False
			is_best_acc = False
			last_epoch = epoch
		writer.close()
		progress_bar.close()
	except KeyboardInterrupt:
		save_checkpoint({
			'epoch': last_epoch,
			'arch': args.arch,
			'image_size': imgsz,
			'state_dict': model.state_dict(),
			'optimizer' : optimizer.state_dict(),
			'nrof_classes': nrof_classes},
			saved_path=weight_save_dir, is_best_loss=False, is_best_acc=False)
		writer.close()
		progress_bar.close()
	
if __name__ == '__main__':
	args = parser_args()
	train(args)