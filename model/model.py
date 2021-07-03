import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, kernels):
        super(MultiScaleCNN, self).__init__()
        self.input_dim = input_dim
        self.kernels = kernels
        self.stream_1 = nn.Conv1d(in_channels = self.input_dim, out_channels=64, kernel_size = self.kernels[0], stride=2)
        self.stream_2 = nn.Conv1d(in_channels = self.input_dim, out_channels=64, kernel_size = self.kernels[1], stride=2)
        self.stream_3 = nn.Conv1d(in_channels = self.input_dim, out_channels=64, kernel_size = self.kernels[2], stride=2)
        self.stream_4 = nn.Conv1d(in_channels = self.input_dim, out_channels=64, kernel_size = self.kernels[3], stride=2)
        self.maxpool = nn.AdaptiveMaxPool1d ((1))
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.relu = ReLU()
        
    def forward(self, x):
        s1_out = self.relu(self.stream_1(x))
        s2_out = self.relu(self.stream_2(x))
        s3_out = self.relu(self.stream_3(x))
        s4_out = self.relu(self.stream_4(x))
        s1_out_avg, s1_out_max, s1_out_std = self.avgpool(s1_out).squeeze(2), self.maxpool(s1_out).squeeze(2), torch.std(s1_out, dim=2)
        s2_out_avg, s2_out_max, s2_out_std = self.avgpool(s2_out).squeeze(2), self.maxpool(s2_out).squeeze(2), torch.std(s2_out, dim=2)
        s3_out_avg, s3_out_max, s3_out_std = self.avgpool(s3_out).squeeze(2), self.maxpool(s3_out).squeeze(2), torch.std(s3_out, dim=2)
        s4_out_avg, s4_out_max, s4_out_std = self.avgpool(s4_out).squeeze(2), self.maxpool(s4_out).squeeze(2), torch.std(s4_out, dim=2)
        
        return s1_out, s2_out, s3_out, s4_out, s1_out_avg, s1_out_max, s1_out_std, s2_out_avg, s2_out_max, s2_out_std,s3_out_avg, s3_out_max, s3_out_std,s4_out_avg, s4_out_max, s4_out_std


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.softmax = nn.Softmax(dim=2)
       
    def forward(self, query, keys):
        keys = keys.permute(2,0,1)
        values = keys
        query = query.unsqueeze(1) # [B,Q] -> [B,1,Q]
        keys = keys.permute(1,2,0) # [T,B,K] -> [B,K,T]
        energy = torch.bmm(query, keys) # [B,1,Q]*[B,K,T] = [B,1,T]
        energy = self.softmax(energy)
        # weight values
        values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
        scaled_val = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
        return scaled_val


    
class MultiScaleCNNSPU(nn.Module):
    def __init__(self,num_classes ):
        super(MultiScaleCNNSPU, self).__init__()
        self.num_classes = num_classes
        self.audio_encoder = MultiScaleCNN(input_dim=96, kernels = [5,7,9,11])
        self.text_encoder = MultiScaleCNN(input_dim=300, kernels = [2,3,4,5])
        self.attention_layer = Attention()
        self.classifier = nn.Sequential(
                                        nn.Linear(4992, 256),
                                        nn.ReLU(),
                                        nn.Linear(256,self.num_classes)
                                )
        self.loss_fun = nn.CrossEntropyLoss()
        
        
    def compute_loss(self, predictions,targets):
        loss = self.loss_fun(predictions, targets)
        return loss
        
    def forward(self, audio_feats, text_feats, x_vector, targets):
        s1,s2,s3,s4, s1_avg, s1_max, s1_std, s2_avg, s2_max, s2_std, s3_avg, s3_max, s3_std, s4_avg, s4_max, s4_std = self.audio_encoder(audio_feats)
        t1,t2,t3,t4, t1_avg, t1_max, t1_std, t2_avg, t2_max, t2_std, t3_avg, t3_max, t3_std, t4_avg, t4_max, t4_std = self.text_encoder(text_feats)
        
        s1_avg_t1 = self.attention_layer(s1_avg, t1)
        s1_avg_t2 = self.attention_layer(s1_avg, t2)
        s1_avg_t3 = self.attention_layer(s1_avg, t3)
        s1_avg_t4 = self.attention_layer(s1_avg, t4)
        
        s2_avg_t1 = self.attention_layer(s2_avg, t1)
        s2_avg_t2 = self.attention_layer(s2_avg, t2)
        s2_avg_t3 = self.attention_layer(s2_avg, t3)
        s2_avg_t4 = self.attention_layer(s2_avg, t4)
        
        s3_avg_t1 = self.attention_layer(s3_avg, t1)
        s3_avg_t2 = self.attention_layer(s3_avg, t2)
        s3_avg_t3 = self.attention_layer(s3_avg, t3)
        s3_avg_t4 = self.attention_layer(s3_avg, t4)
        
        s4_avg_t1 = self.attention_layer(s4_avg, t1)
        s4_avg_t2 = self.attention_layer(s4_avg, t2)
        s4_avg_t3 = self.attention_layer(s4_avg, t3)
        s4_avg_t4 = self.attention_layer(s4_avg, t4)
        
        #################################################
        s1_max_t1 = self.attention_layer(s1_max, t1)
        s1_max_t2 = self.attention_layer(s1_max, t2)
        s1_max_t3 = self.attention_layer(s1_max, t3)
        s1_max_t4 = self.attention_layer(s1_max, t4)
        
        s2_max_t1 = self.attention_layer(s2_max, t1)
        s2_max_t2 = self.attention_layer(s2_max, t2)
        s2_max_t3 = self.attention_layer(s2_max, t3)
        s2_max_t4 = self.attention_layer(s2_max, t4)
        
        s3_max_t1 = self.attention_layer(s3_max, t1)
        s3_max_t2 = self.attention_layer(s3_max, t2)
        s3_max_t3 = self.attention_layer(s3_max, t3)
        s3_max_t4 = self.attention_layer(s3_max, t4)
        
        s4_max_t1 = self.attention_layer(s1_max, t1)
        s4_max_t2 = self.attention_layer(s1_max, t2)
        s4_max_t3 = self.attention_layer(s1_max, t3)
        s4_max_t4 = self.attention_layer(s1_max, t4)
        
        ###################################################
        s1_std_t1 = self.attention_layer(s1_std, t1)
        s1_std_t2 = self.attention_layer(s1_std, t2)
        s1_std_t3 = self.attention_layer(s1_std, t3)
        s1_std_t4 = self.attention_layer(s1_std, t4)
        
        s2_std_t1 = self.attention_layer(s2_std, t1)
        s2_std_t2 = self.attention_layer(s2_std, t2)
        s2_std_t3 = self.attention_layer(s2_std, t3)
        s2_std_t4 = self.attention_layer(s2_std, t4)
        
        s3_std_t1 = self.attention_layer(s3_std, t1)
        s3_std_t2 = self.attention_layer(s3_std, t2)
        s3_std_t3 = self.attention_layer(s3_std, t3)
        s3_std_t4 = self.attention_layer(s3_std, t4)
        
        s4_std_t1 = self.attention_layer(s1_std, t1)
        s4_std_t2 = self.attention_layer(s1_std, t2)
        s4_std_t3 = self.attention_layer(s1_std, t3)
        s4_std_t4 = self.attention_layer(s1_std, t4)
        
        ######################################################
        S = torch.cat((s1_avg_t1,s1_avg_t2,s1_avg_t3,s1_avg_t4, s2_avg_t1,s2_avg_t2,s2_avg_t3,s2_avg_t4, s3_avg_t1,s3_avg_t2,s3_avg_t3,s3_avg_t4, s4_avg_t1,s4_avg_t2,s4_avg_t3,s4_avg_t4,
                      s1_max_t1,s1_max_t2,s1_max_t3,s1_max_t4, s2_max_t1,s2_max_t2,s2_max_t3,s2_max_t4, s3_max_t1,s3_max_t2,s3_max_t3,s3_max_t4, s4_max_t1,s4_max_t2,s4_max_t3,s4_max_t4,
                      s1_std_t1,s1_std_t2,s1_std_t3, s1_std_t4,s2_std_t1,s2_std_t2,s2_std_t3,s2_std_t4, s3_std_t1,s3_std_t2,s3_std_t3,s3_std_t4, s4_std_t1,s4_std_t2,s4_std_t3,s4_std_t4),dim=1)
        A = torch.cat((s1_avg, s1_max, s1_std, s2_avg, s2_max, s2_std, s3_avg, s3_max, s3_std, s4_avg, s4_max, s4_std ), dim=1)
        T = torch.cat((t1_std, t2_avg, t2_max, t2_std, t3_avg, t3_max, t3_std, t4_avg, t4_max, t4_std), dim=1)
        final_vec = torch.cat((S, A, T, x_vector), dim=1)
        pred = self.classifier(final_vec)
        loss = self.compute_loss(pred, targets)
        return loss, pred
    