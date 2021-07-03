import os
import argparse
import glob
import yaml
import numpy as np
import torch


import logging
import os
import re

import yaml
import torch


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin)
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    print(infos)
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)
        
        
def average_models(src_path, dst_model, val_best=True, num=5, min_epoch=0, max_epoch=10000):
    checkpoints = []
    val_scores = []
    if val_best:
        yamls = glob.glob('{}/[!train]*.yaml'.format(src_path))
        for y in yamls:
            with open(y, 'r') as f:
                dic_yaml = yaml.load(f)
                loss = dic_yaml['cv_loss']
                epoch = dic_yaml['epoch']
                if epoch >= min_epoch and epoch <= max_epoch:
                    val_scores += [[epoch, loss]]
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::1]
        print("best val scores = " + str(sorted_val_scores[:num, 1]))
        print("selected epochs = " +
              str(sorted_val_scores[:num, 0].astype(np.int64)))
        path_list = [
            src_path + '/{}.pt'.format(int(epoch))
            for epoch in sorted_val_scores[:num, 0]
        ]
    else:
        path_list = glob.glob('{}/[!avg][!final]*.pt'.format(src_path))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-num:]
    print(path_list)
    avg = None
    num = num
    assert num == len(path_list)
    for path in path_list:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(dst_model))
    torch.save(avg, dst_model)