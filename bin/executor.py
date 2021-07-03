import logging
from math import log
from librosa.core import audio
import torch
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
import tqdm

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, data_loader, device, writer,):
        ''' Train one epoch
        '''
        model.train()
        clip =  50.0
        log_interval =  10
        accum_grad = 1
        rank=0
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        for batch_idx, (audio_feats,text_feats, x_vectors, target) in enumerate(data_loader):
            audio_feats = audio_feats.to(device)
            text_feats = text_feats.to(device)
            x_vectors = x_vectors.to(device)
            target = target.squeeze(-1).to(device)
            num_utts = target.size(0)
            if num_utts == 0:
                continue
            context = None

            loss, predictions = model(audio_feats, text_feats.permute(0,2,1), x_vectors, 
                                        target)
            loss = loss / accum_grad
            loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(torch.Tensor([grad_norm])):
                    optimizer.step()
                optimizer.zero_grad()
                self.step += 1
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                print('TRAIN Batch {}/{} loss {:.6f} lr {:.8f} rank {}'.format(
                                  batch_idx, num_total_batch,
                                  loss.item(), lr, rank))

    def cv(self, model, data_loader, device):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = 10
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        gt_labels , pred_labels = list(), list()
        with torch.no_grad():
            for batch_idx, (audio_feats,text_feats, x_vectors, target) in enumerate(data_loader):
                audio_feats = audio_feats.to(device)
                text_feats = text_feats.to(device)
                x_vectors = x_vectors.to(device)
                target = target.squeeze(-1).to(device)
                num_utts = target.size(0)
                if num_utts == 0:
                    continue
                loss, predictions = model(audio_feats, text_feats.permute(0,2,1), x_vectors, 
                                          target)
                gt_labels.extend(list(target.cpu().numpy()))
                pred_labels.extend(list(predictions.cpu().numpy()))
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    print('CV Batch {}/{} loss {:.6f} history loss {:.6f}'.format(
                                      batch_idx, num_total_batch, loss.item(),
                                      total_loss / num_seen_utts))

        
        
        return total_loss, num_seen_utts
        
    
    def evaluation(self, model, data_loader, device):
        ''' Evaluation
        '''
        model.eval()
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        gt_labels , pred_labels = list(), list()
        with torch.no_grad():
            for  batch_idx, (audio_feats,text_feats, x_vectors, target) in enumerate(data_loader):
                audio_feats = audio_feats.to(device)
                text_feats = text_feats.to(device)
                x_vectors = x_vectors.to(device)
                target = target.squeeze(-1).to(device)
                assert audio_feats.shape[0]==1
                num_utts = target.size(0)
                if num_utts == 0:
                    continue
                _, predictions = model(audio_feats, text_feats.permute(0,2,1), x_vectors, 
                                          target)
                predictions = torch.topk(predictions,1)[1][0].item()
                gt_labels.append(int(target[0].cpu().numpy()))
                pred_labels.append(int(predictions))
        test_acc = accuracy_score(gt_labels, pred_labels)
        return test_acc