'''
Test efficient-net trained model 
Visualize result to excel file.

author: phatnt
date: May-01-2022
'''
import os
import csv
import cv2
import sys
import shutil
import argparse
import xlsxwriter
import numpy as np

import torch
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm
from custom_dataset import CustomDataset, Normalize, ToTensor, Resize

sys.path.append('./efficientnet')
from model import EfficientNet

def parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, required=True)
	parser.add_argument('--weight', type=str, required=True)
	parser.add_argument('--thres', type=str, default=None)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=32)
	return parser.parse_args()

def softmax(x):
	'''
		Calculate softmax of an array.
		Args:
			x: an array.
		Return:
			Softmax score of input array.
	'''
	assert x is not None, '[ERROR]: input is none!'
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0)

def get_cofusion_matrix(evaluate_dict):
	'''
		Get confusion matrix from test set.
		Args:
			evaluate_dict: dictionary for evaluate include:
				+ model: testing model.
				+ train_classes: trained classes.
				+ device: testing device.
				+ threshold: class threshold array.
				+ data_loader: torch data_loader format.
		Return:
			accuracy: total accuracy base on test set.
			result_matrix: confusion matrix
			false_defects: a list of false_defects to generate image in report file/
			underkill: number of underkill images.
			overkill:: number of overkill images.
	'''
	assert evaluate_dict is not None, '[ERROR]: evaluate dict is None!'
	model = evaluate_dict['model']
	model.eval()
	train_classes = evaluate_dict['train_classes']
	num_classes = len(train_classes)

	result_matrix = np.zeros((num_classes, num_classes + 1), dtype=np.int32) 
	overkill = 0
	underkill = 0
	false_defects = []
	new_thresholds = np.ones(num_classes)
	process_bar = tqdm()

	# Caculate confusion matrix
	with torch.no_grad():
		for i, sample_batched in enumerate(evaluate_dict['data_loader']):
			# Get output from model
			images = sample_batched['image']
			paths = sample_batched['path']
			targets = sample_batched['target']
			images = images.to(device=evaluate_dict['device'], dtype=torch.float)
			targets = np.array(targets)
			score = model(images)
			score = score.cpu().data.numpy()

			# Compare with target
			for i in range (len(targets)):
				gt_label = targets[i]
				softmaxed = softmax(score[i])
				highest_score_idx = softmaxed.argmax()
				# Apply thesholds
				mask = (softmaxed >= evaluate_dict['thresholds'])
				predicted_label = num_classes if not any(mask) else (softmaxed * mask).argmax()
				# Cofusion matrix
				result_matrix[gt_label][predicted_label] += 1
				if gt_label != predicted_label: # False predict
					false_defect = {'image_path': paths[i],
									'gt_label': gt_label,
									'predicted_label': predicted_label,
									'gt_score': softmaxed[gt_label],
									'highest_label': highest_score_idx,
									'highest_score': softmaxed[highest_score_idx]}
					false_defects.append(false_defect)
					if predicted_label < num_classes:
						if train_classes[gt_label] == 'Pass':
							overkill += 1
						if train_classes[predicted_label] == 'Pass':
							underkill += 1
					process_bar.set_description(f'Number of false predict cases: {len(false_defects)}')
				elif gt_label == predicted_label: # True predict
					if (softmaxed[gt_label] < new_thresholds[gt_label]):
						new_thresholds[gt_label] = softmaxed[gt_label]
	process_bar.close()
	# Calculate accuracy
	count = 0
	for i in range(num_classes):
		count+=result_matrix[i][i]
	accuracy = (count/np.sum(result_matrix))*100
	
	print('New thresholds:', new_thresholds)
	return accuracy, result_matrix, false_defects, underkill, overkill

def calculate_exel_char(start_char, length):
	'''
		Caculate column character(s) of excel in order to expand the width of those column.
		Args:
			start_char: starting column.
			length: number of columns that need to be expand width.
		Return:
			pos: excel's column character(s).
	'''
	start_order = ord(start_char)
	end_order = start_order + length
	over = False
	first_char_ord = ord('@')
	while (end_order > ord('Z')):
		over = True
		first_char_ord += 1
		end_order -= 26
	if over:
		pos =f'{chr(first_char_ord)}{chr(end_order)}'
	else:
		pos = chr(end_order)
	return pos

def set_cols(sheet, list_col_width):
	'''
		Set column(s) width from calculated char.
		Args:
			sheet: excel sheet.
			list_col_width: list column(s) and their width.
	'''
	for col,width in list_col_width:
		if len(col) == 1:
			sheet.set_column(f'{col}:{col}', width)
		elif len(col) > 1:
			sheet.set_column(f'{col}', width)

def create_report(test_file, evaluate_dict, report_save_dir='./'):
	'''
	Create report sheets.
		+ Sheet 1: Confusion matrix.
		+ Sheet 2: Underkill images.
	Args:

	'''
	# Get test set quantity
	train_classes = evaluate_dict['train_classes']
	num_classes = len(train_classes)
	thresholds = evaluate_dict['thresholds']
	test_quantity = np.zeros(num_classes, dtype=np.int32)
	with open(test_file, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		label = int(line.split("\"")[-1].strip())
		test_quantity[label] += 1
	total_accuracy, result_matrix, false_defects, underkill, overkill = get_cofusion_matrix(evaluate_dict)

	# REPORT
	report_save_path = os.path.join(report_save_dir, 'report.xlsx')
	report_sheet = xlsxwriter.Workbook(report_save_path)
	sheet1 = report_sheet.add_worksheet('TOMOC Report')
	sheet2 = report_sheet.add_worksheet('False Predict')
	HEADER = report_sheet.add_format({'bold':1, 'border':1, 'align':'center', 'valign':'vcenter', 'fg_color':'#B9CDE5'})
	HEADER.set_text_wrap()
	NORMAL = report_sheet.add_format({'border':1, 'align':'center', 'valign':'vcenter'})
	NORMAL.set_text_wrap()

	# SHEET 1
	# sheet 1 fotmat
	sheet1_column_width = [ ('B', 15), # Model's name
							('C', 20), # Defect's name
							('D', 12), # Quantities
							('E:{}'.format(calculate_exel_char('E', num_classes-1)), 12), # Train classes
							('{}:{}'.format(calculate_exel_char('E', num_classes), 
											calculate_exel_char('E', num_classes+2)), 12)] # Thresholds, Accuracy

	set_cols(sheet1, sheet1_column_width)
	sheet1.set_row(2, 50)
	sheet1.set_row(3+num_classes, 50)

	for row in range(1, 4):
		for col in range(1, 7+num_classes):
			sheet1.write(row, col, '', HEADER)
	#sheet1.freeze_panes(3, 4)
	sheet1.merge_range(f"E2:{calculate_exel_char('E',num_classes-1)}2", 'Defect Type', HEADER)
	sheet1.merge_range(f"B4:B{4+num_classes-1}", 'Efficient-Net', HEADER)

	# Sheet 1 content
	sheet1_header1 = ['Model\'s Name','Defect Type', 'Quantities'] + train_classes + ['Unknown','Threshold','Accuracy']
	sheet1.write_row(2, 1, sheet1_header1, HEADER)
	#sheet1.write_row(1, 6+num_classes-1, row_header_1_2, HEADER)

	for i in range (num_classes):
		# Class name / quantities / result / threshold / acc
		sheet1_content1 = [f'{train_classes[i]}', f'{test_quantity[i]}'] + [str(x) for x in result_matrix[i]] + [f'{thresholds[i]}', f'{result_matrix[i][i]/test_quantity[i]*100:.2f}%']
		sheet1.write_row(3+i, 2, sheet1_content1, NORMAL)
		sheet1.write(3+i, 4+i, result_matrix[i][i], HEADER)
	
	# Sheet1 sumarize result 
	total_test_img = np.sum(test_quantity)
	true_predict = np.sum([result_matrix[i][i] for i in range(num_classes)])
	sheet1_content2 = ['Underkill', underkill, 'Overkill', overkill, 'Total False Predict', total_test_img-true_predict, 'Total Accuracy', f'{total_accuracy:.2f}%']
	sheet1.write_row(3+num_classes ,num_classes-1, sheet1_content2, HEADER)

	# SHEET 2
	sheet2.set_column('C:G', 34)
	sheet2.freeze_panes(2, 2)
	sheet2_header1 = ['No', 'False Defects', 'Ground True', 'Predicted','Image Path' ]
	sheet2.write_row(1, 1, sheet2_header1, HEADER)
	saved_path = "./temp/"
	os.makedirs(saved_path, exist_ok=True)

	for count, false_defect in enumerate(false_defects):
		sheet2.set_row(2+count, 200)
		if false_defect['predicted_label'] < num_classes: 
			predicted_class = train_classes[false_defect['predicted_label']]
			predicted_score = false_defect['highest_score']
			predicted_text = f'{predicted_class.upper()}\n({predicted_score*100:.3f}%)'
		else:
			predicted_class = 'Unknown'
			predicted_text = f'{predicted_class.upper()}'
		gt_class = train_classes[false_defect['gt_label']]
		gt_score = false_defect['gt_score']
		gt_text = f'{gt_class.upper()}\n({gt_score*100:.3f}%)'
		
		sheet2_content1 = [count+1, 'image', gt_text, predicted_text, false_defect['image_path']]
		sheet2.write_row(2+count, 1, sheet2_content1, NORMAL)

		# Insert image
		image = cv2.imread(false_defect['image_path'])
		resized = cv2.resize(image, (250, 250), interpolation=cv2.INTER_AREA)
		saved_img_path = os.path.join(saved_path, f"{count+1}.bmp")
		cv2.imwrite(saved_img_path ,resized)
		sheet2.insert_image(2+count, 2, saved_img_path ,{'x_scale':1.0 , 'y_scale':1.0, 'x_offset': 5, 'y_offset': 5, 'object_position': 1})

	report_sheet.close()
	shutil.rmtree(saved_path)
	print(f'{report_save_path} Created')

def test(args):
	'''
		Evaluating model via test set then save result into excel file.
	'''
	print("Export Excel Report")
	
	label_file =  os.path.join(args.data, "label.txt")
	test_file = os.path.join(args.data, "test.txt")
	assert os.path.isfile(label_file), f'[ERROR] Could not found {label_file}!'
	assert os.path.isfile(test_file), f'[ERROR] Could not found {test_file}!'
	
	# Get Classes name
	thresholds = []
	train_classes = []
	with open(label_file, 'r', encoding='utf-8') as label_f:
		lines = label_f.readlines()
	for line in lines:
		class_name = line.strip()
		train_classes.append(class_name)
	num_classes = len(train_classes)
	# Create threshold file
	if args.thres is None:
		thres_file = os.path.join(os.path.join(args.data, 'thres.csv'))
		csv_file = open(thres_file, 'w', encoding='UTF8')
		csv_writer = csv.writer(csv_file)
		for class_name in train_classes:
			csv_writer.writerow([f'{class_name}','0.2'])
		csv_file.close()
		print('[INFO]: {} created!'.format(thres_file))
	else:
		assert os.path.isfile(args.thres), f'[ERROR] Could not found {args.thres}'
		thres_file = args.thres

	# Get thresholds
	print('[INFO] Train classes / Threshold:')
	with open(thres_file, 'r', encoding='utf-8') as thres_f:
		csv_reader = csv.reader(thres_f, delimiter = ',')
		for row in csv_reader:
			print(row)
			threshold = float(row[1])
			thresholds.append(threshold)
	assert len(thresholds) == num_classes

	# Load model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(args.device)
	checkpoint = torch.load(args.weight, map_location=device)
	imgsz = checkpoint['image_size']
	model = EfficientNet.from_name(f'efficientnet-b{checkpoint["arch"]}', num_classes=checkpoint['num_classes'], 
									image_size=imgsz, in_channels=checkpoint["in_channels"])

	data_transforms = transforms.Compose([Resize(imgsz), Normalize(), ToTensor()])
	data_loader = torch.utils.data.DataLoader(CustomDataset(test_file, data_transforms),
												batch_size=args.batch_size, shuffle=False,
												num_workers=args.workers, pin_memory=False)

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
	create_report(test_file=test_file, evaluate_dict=evaluate_dict, report_save_dir=os.path.dirname(args.weight))

if __name__ == '__main__':
	args = parser_args()
	test(args)