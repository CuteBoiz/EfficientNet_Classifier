"""
Generate data from bounding boxes of csv files.
Crop then save into specific defect's name folder.

author: phatnt
date: 2021-08-10

"""

import cv2
import os
from datetime import datetime
import glob
import csv
import json
from math import cos, sin, atan2, sqrt
import numpy as np

DEFECT_ORIGIN_IMG_FOLDER = '/mnt/sdb2/home/bhabani/Pravin/dataset_backup/BV Training Data_Image validated/Image_29042021_122748/'
PASS_ORIGIN_IMG_FOLDER = '/mnt/sdb2/home/bhabani/Pravin/dataset_backup/pass_image/LS3 BV FOK'
TRAIN_CLASS = ["AB","BCE","CE","CT","DB","DC","DoE","EG","EH","EHaCM","ET","IB","LT","MC","MF","PI","SB","SI","SwC","WS"]
IMAGE_SIZE = 128
	
def generate_data(csv_file_paths, saved_root_dir):
	'''
	Get defect from csv_files. Crop them from ORIGIN_IMAGE_FOLDERS.
	Then save them into specific class-folder. 

	Args:
		csv_file_paths: list of csv files
		saved_root_dir: Cropped dataset saved place.
	'''
	#Read CSV files
	for csv_file in csv_file_paths:
		with open(csv_file, 'r') as f_csv:
			csv_reader = csv.reader(f_csv, delimiter = ',')
			_ = next(csv_reader)
			_ = next(csv_reader)
			for row in csv_reader:
				#Get information of each defect
				image_name = row[0]
				if image_name.rfind('SBV16') != -1:
					image_path = os.path.join(PASS_ORIGIN_IMG_FOLDER, image_name)
				else:
					image_path = os.path.join(DEFECT_ORIGIN_IMG_FOLDER, image_name)
				rect = json.loads(row[2])
				#PASS IMAGES
				if rect == {}: 
					image = cv2.imread(image_path)
					ori_h, ori_w = image.shape[:2]
					count = 0
					#Generate a sliding window to crop Pass image
					x_range = np.arange(start=0, stop=ori_w-IMAGE_SIZE, step=IMAGE_SIZE)
					y_range = np.arange(start=0, stop=ori_h-IMAGE_SIZE, step=IMAGE_SIZE)
					for x in x_range:
						for y in y_range:
							#Make sure the black area is smaller than 30%
							cropped = image[y:y+IMAGE_SIZE, x:x+IMAGE_SIZE]
							gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
							black_pixel = np.sum(gray <= 51)
							if (black_pixel/(IMAGE_SIZE*IMAGE_SIZE)) < 0.3:
								save_dir = save_dir = os.path.join(saved_root_dir, "Pass")
								os.makedirs(save_dir, exist_ok=True)

								count+=1
								save_img_name = "{}-Pass_{:03n}.{}.bmp".format(image_name.split("-")[0], count, image_name.split(".")[-2])
								saved_img_path = os.path.join(save_dir, save_img_name)
								cv2.imwrite(saved_img_path, cropped)
								print(saved_img_path)
							else:
								continue
				#DEFECT IMAGES
				else:
					try:
						x = int(rect['x'])
						y = int(rect['y'])
						w = int(rect['width'])
						h = int(rect['height'])
					except:
						print(csv_file)
						print(row)
						continue
					defect_type = row[3].split("\"")[-2]
					#Read Image & Crop
					image = cv2.imread(image_path)
					ori_h, ori_w = image.shape[:2]

					if w > IMAGE_SIZE and h > IMAGE_SIZE:
						cropped = image[y:y+h,x:x+w]
						resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
					else:
						x_center = x+w/2
						y_center = y+h/2
						box_x = int(x_center - IMAGE_SIZE/2)
						box_y = int(y_center - IMAGE_SIZE/2)
						cropped = image[max(box_y, 0):min(box_y+IMAGE_SIZE, ori_h), max(box_x, 0):min(box_x+IMAGE_SIZE, ori_w)]
						resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
					
					#Find valid filename & Save Image
					save_dir = os.path.join(saved_root_dir, defect_type)
					os.makedirs(save_dir, exist_ok=True)
					count = 1
					saved_path = ''
					while (1):
						saved_name = "{}-{}_{:02n}.{}.bmp".format(image_name.split("-")[0], defect_type, count, image_name.split(".")[-2])
						saved_path = os.path.join(save_dir, saved_name)
						if not os.path.isfile(saved_path):
							break
						count+=1
				
					print(saved_path)
					cv2.imwrite(saved_path, resized)
		

def split_train_valid_test(dataset_folder, ratio=[0.8, 0.1, 0.1]):
	'''
		Split data set into train/test/valid and create label.csv
		Args:
			dataset_folder: dataset folder |
										   |_class_name_1/images
										   |_class_name_2/images
										   |_...
			ratio: train/valid/test ratio
	'''
	assert os.path.isdir(dataset_folder), '[ERROR] Could not found {}'.format(dataset_folder)
	assert isinstance(ratio, list), '[ERROR] ratio must be a list'
	assert len(ratio) == 3, '[ERROR] ratio lenght must = 3'
	assert int(sum(ratio)) == 1, '[ERROR] ratio sum must be equal to 1'

	save_folder = "./data_{}/".format(datetime.today().strftime('%Y-%m-%d'))
	os.makedirs(save_folder, exist_ok=True)

	train_file = os.path.join(save_folder, "train.txt")
	test_file = os.path.join(save_folder, "test.txt")
	valid_file = os.path.join(save_folder, "valid.txt")
	label_file = os.path.join(save_folder, "label.csv")

	#Create Label File
	with open(label_file, 'w', encoding='UTF8') as csv_file:
		writer = csv.writer(csv_file)
		count = 0
		with open('/home/phatnt/Alcon_Project/csv_files/all_defect.txt', 'r') as f:
			lines = f.readlines()
		for line in lines:
			line = line.strip()
			label, label_code = line.split("\"")[1], line.split("\"")[2].strip()
			if label_code in TRAIN_CLASS:
				writer.writerow([label,label_code,count])
				count+=1
		writer.writerow(['Pass','Pass',count])
	
	#Load Defect dictionary
	defect_dict = {}
	with open(label_file, 'r', encoding='UTF8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		for row in csv_reader:
			defect_dict[row[1]] = row[2]
	print(defect_dict)

	#Read Defect Data
	dataset = []
	for defect in TRAIN_CLASS:
		image_paths = glob.glob(os.path.join(dataset_folder, defect, "*.bmp"))
		label_ids = np.ones(len(image_paths), dtype=np.int32)*int(defect_dict[defect])
		dataset.extend((zip(image_paths, label_ids)))
	print(len(dataset))

	#Read Pass Data
	pass_image_paths = glob.glob(os.path.join(dataset_folder, 'Pass', "*.bmp"))
	if len(pass_image_paths) > len(dataset):
		pass_idx = np.arange(len(pass_image_paths))
		np.random.shuffle(pass_idx)
		pass_paths =  [pass_image_paths[pass_idx[i]] for i in range(len(dataset))]
	else:
		pass_paths = pass_image_paths
	label_ids = np.ones(len(pass_paths), dtype=np.int32)*int(defect_dict["Pass"])
	dataset.extend((zip(pass_paths, label_ids)))
	print(len(dataset))

	#Shuffle data
	index = np.arange(len(dataset))
	np.random.shuffle(index)
	train_idx = int(ratio[0] * len(dataset))
	valid_idx = int(train_idx + ratio[1] * len(dataset))
	train_set = [dataset[index[i]] for i in range(train_idx)]
	valid_set = [dataset[index[i]] for i in range(train_idx, valid_idx)]
	test_set = [dataset[index[i]] for i in range(valid_idx, len(dataset))]

	print(len(train_set), len(valid_set), len(test_set))

	with open(train_file, 'w') as train_f:
		for data in train_set:
			train_f.write("\"{}\" {}\n".format(data[0], data[1]))

	with open(valid_file, 'w') as valid_f:
		for data in valid_set:
			valid_f.write("\"{}\" {}\n".format(data[0], data[1]))

	with open(test_file, 'w') as test_f:
		for data in test_set:
			test_f.write("\"{}\" {}\n".format(data[0], data[1]))


if __name__ == '__main__':
	csv_file_paths = glob.glob(os.path.join('/home/phatnt/Alcon_Project/csv_files', "*.csv"))
	save_root_dir = '/mnt/sdb2/home/leonard/Alcon_Dataset/{}/'.format(datetime.today().strftime('%Y-%m-%d'))
	#generate_data(csv_file_paths, save_root_dir)
	split_train_valid_test(dataset_folder=save_root_dir)
