'''
[Train] - [Export excel] - [Export onnx] for Alcon SPC classifier.

author: phatnt
date modify: 2022-01-07
'''
import cv2
import argparse
import os
import glob 
import numpy as np

from efficientnet_pytorch import EfficientNet
from custom_dataset import CustomDataset, Normalize, ToTensor, Resize
from utils.train_utilities import  get_class_weight, train_eff
from utils.report_utilities import create_report, softmax, infer_preprocess

import torch
import torchvision.transforms as transforms

def prepare_arguments():
    """
    Prepare Arguments
    [Train]: Train Efficient Net.

    [Export ONNX]: Export onnx model from trained model.

    [Export Excel Report]: Export report.

    [Infer]: Model inference.
    """
    
    subparser = parser.add_subparsers(dest="mode")

    train_parser = subparser.add_parser("train")
    
    



    

    

    
    



def export_excel(args):
    



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
