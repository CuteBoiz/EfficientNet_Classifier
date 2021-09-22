'''
Generate txt files from Alcon Classification data.

author: phatnt
date modify: 2021-08-25
'''


import os
import glob
import numpy as np
from datetime import datetime

DATASET_FOLDER = '/mnt/sdb2/home/leonard/Alcon_Dataset/Classification_Data/'

TRAIN_CLASSES =  ['Field of view off center',
			   	  'Lens not completely set in cuvette', 
			   	  'Lens off center in cuvette',
			   	  'Multiple contact lenses',
			   	  'No Lens',
			   	  'Non Circular Lenses',
			   	  'Roadmaps',
			  	  'Underdosed PVA',
			   	  'Water on cuvette',
			   	  'Pass']

def main(ratio=[0.8, 0.1, 0.1]):
	assert isinstance(ratio, list), '[ERROR] ratio must be a list'
	assert len(ratio) == 3, '[ERROR] ratio lenght must = 3'
	assert int(sum(ratio)) == 1, '[ERROR] ratio sum must be equal to 1'
	
	save_folder = "./data_{}/".format(datetime.today().strftime('%Y-%m-%d'))
	os.makedirs(save_folder, exist_ok=True)
	train_file = os.path.join(save_folder, "train.txt")
	test_file = os.path.join(save_folder, "test.txt")
	valid_file = os.path.join(save_folder, "valid.txt")
	label_file = os.path.join(save_folder, "label.txt")

	#Create label file
	with open(label_file, 'w', encoding='UTF8') as label_f:
		for class_name in TRAIN_CLASSES:
			data_path = os.path.join(DATASET_FOLDER, class_name)
			assert os.path.isdir(data_path), "[ERROR] Could not found {}".format(data_path)
			label_f.write('{}\n'.format(class_name))
	print('[INFO]: Label file {} created!'.format(label_file))
	raise Exception
	dataset = []
	for class_name in TRAIN_CLASSES:
		data_path = os.path.join(DATASET_FOLDER, class_name)
		img_paths = glob.glob(os.path.join(data_path, '*.bmp'))
		label_ids = np.ones(len(img_paths), dtype=np.int32)*TRAIN_CLASSES.index(class_name)
		dataset.extend((zip(img_paths, label_ids)))
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
	main()