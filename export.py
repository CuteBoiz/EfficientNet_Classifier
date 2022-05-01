'''
Export efficient-net model to onnx engine

author: phatnt
date: May-01-2022
'''
import os
import torch
import argparse
from efficientnet_pytorch import EfficientNet

def parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--weight", type=str, required=True)
	parser.add_argument("--batch", type=int, default=1)
	parser.add_argument("--name", type=str, default=None)
	parser.add_argument("--opset", type=int, default=9)
	parser.add_argument("--verbose", action='store_true', default=False)
	return parser.parse_args()

def export(args):
    '''
        Export pytorch weight to onnx engine.
    '''
    print("[INFO] Exporting {args.weight} to Onnx engine.")
    assert args.max_batch_size > 0
    assert os.path.isfile(args.weight), f'[ERROR] {args.weight} not found!'

    checkpoint = torch.load(args.weight, map_location='cpu')
    image_size = checkpoint['image_size']
    model = EfficientNet.from_name(f'efficientnet-b{checkpoint["arch"]}', num_classes=checkpoint['nrof_classes'],
                                    image_size=image_size, in_channels=checkpoint["in_channels"])
    
    print('[INFO] Model info:')
    print(f'\t + Architecture:  EfficientNet-b{checkpoint["arch"]}')
    print(f'\t + Input shape: {args.max_batch_size} x 3 x {image_size} x {image_size}')
    print(f'\t + Ouput shape: {checkpoint["nrof_classes"]}')

    model.load_state_dict(checkpoint['state_dict'])
    model.set_swish(memory_efficient=False)
    model.eval()
    dummy_input = torch.randn(args.max_batch_size, 3, image_size, image_size, requires_grad=True)
    saved_name = args.name if args.name is not None else args.weight.replace('.pth', '.onnx')
    torch.onnx.export(model, dummy_input, saved_name,
                    opset_version = args.opset,
                    verbose=args.verbose, export_params=True, do_constant_folding=True)
    
    print(f'[INFO]: {saved_name} created!, Exporting Done!')

if __name__ == '__main__':
	args = parser_args()
	export(args)