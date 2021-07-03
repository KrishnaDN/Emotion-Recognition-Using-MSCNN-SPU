import os
import numpy as np
from model.model import MultiScaleCNNSPU
from dataset.iemocap import IEMOCAPDataset
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-pickle_filepath',type=str,default='/media/newhd/IEMOCAP_dataset/IEMOCAP_x_vector.pickle')
    parser.add_argument('-glove_filepath',type=str, default='/media/newhd/IEMOCAP_dataset/glove.840B.300d.pickle')
    parser.add_argument('-audio_feat_dim', action="store_true", default=96)
    parser.add_argument('-text_feat_dim', action="store_true", default=300)
    parser.add_argument('-num_classes', action="store_true", default=4)
    parser.add_argument('-batch_size', action="store_true", default=32)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    args = parser.parse_args()
    

def main():
    args = arg_parser()
    print(args)
    
if __name__=='__main__':
    main()