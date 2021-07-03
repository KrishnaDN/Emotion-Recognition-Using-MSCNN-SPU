import os
import numpy as np
from model.model import MultiScaleCNNSPU
from dataset.iemocap import IEMOCAPDataset, TrainTest
import argparse
from bin.executor import Executor
from utils.model_utils import save_checkpoint, load_checkpoint, average_models
from torch.utils.data import DataLoader  
import torch
from torch import optim
import logging

def arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-pickle_filepath',type=str,default='/media/newhd/IEMOCAP_dataset/IEMOCAP_x_vector.pickle')
    parser.add_argument('-glove_filepath',type=str, default='/media/newhd/IEMOCAP_dataset/glove.840B.300d.pickle')
    parser.add_argument('-model_dir',type=str, default='experiments')
    
    parser.add_argument('-audio_feat_dim', action="store_true", default=96)
    parser.add_argument('-text_feat_dim', action="store_true", default=300)
    parser.add_argument('-num_classes', action="store_true", default=4)
    parser.add_argument('-batch_size', action="store_true", default=32)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-num_epochs', action="store_true", default=100)
    
    args = parser.parse_args()
    return args



def main():
    args = arg_parser()
    train_test_split = TrainTest(args.pickle_filepath, 3)
    train_list, test_list = train_test_split()
    trainset = IEMOCAPDataset(train_list, args.glove_filepath)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    testset = IEMOCAPDataset(test_list, args.glove_filepath)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    model = MultiScaleCNNSPU(num_classes=4)
    use_cuda = args.use_gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model_dir = args.model_dir
    writer = None
    executor = Executor()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
    infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    if start_epoch == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)
        
    for epoch in range(start_epoch, args.num_epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, train_loader, device,
                    writer)
        
        test_acc = executor.evaluation(model, test_loader, device)
        
        print('Epoch {} CV info accuracy {}'.format(epoch, test_acc))
        
        save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
        save_checkpoint(
            model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'step': executor.step,
                'test_acc': float(test_acc)
            })
        

    
    
if __name__=='__main__':
    main()