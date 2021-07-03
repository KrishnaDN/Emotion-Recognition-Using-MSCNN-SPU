import os
import torch.nn as nn
import pickle
import random
import librosa
import numpy as np
import torch

LABELS = {'hap':0,'ang':1, 'exc':0,'sad':2,'neu':3}
FOLDS = {1:'Ses01M',2:'Ses01F', 3:'Ses02M',4:'Ses02F', 5:'Ses03M',6:'Ses03F', 7:'Ses04M',8:'Ses04F', 9:'Ses05M',10:'Ses05F'}

class TrainTest:
    def __init__(self, dataset_file, fold_id):
        self.fold = FOLDS[fold_id]
        with open(dataset_file, 'rb') as handle:
            self.data = pickle.load(handle)

    def __call__(self,):
        filenames = list(self.data.keys())
        
        train_splits, test_splits = list(),list()
        for filename in filenames:
            if filename.split('_')[0]==self.fold:
                test_splits.append(self.data[filename])
            else:
                train_splits.append(self.data[filename])
        return train_splits, test_splits
        
class IEMOCAPDataset:
    def __init__(self, dataset, glove_path, max_frames=750, max_words=50):
        self.dataset = dataset
        self.glove_path = glove_path
        self.max_frames = max_frames
        self.max_words = max_words
        self._load_glove
        
    def _feature_extraction(self,audio_path):
        wav, _ = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(wav, sr=16000, n_mfcc=32, win_length = 400, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        double_delta = librosa.feature.delta(delta)
        features = np.concatenate((mfcc, delta, double_delta), axis=0)
        if features.shape[1]>=self.max_frames:
            return features[:,:self.max_frames]
        else:
            padded_feats = np.concatenate((features, np.zeros((features.shape[0], self.max_frames-features.shape[1]))),axis=1)
            assert padded_feats.shape[1]==self.max_frames
            return padded_feats
    
    @property
    def _load_glove(self,):
        self.word2vec = {}
        with open(self.glove_path, 'rb') as handle:
            self.word2vec = pickle.load(handle)

    def _get_glove_features(self,text):
        import re
        clean_string = ' '.join(list(filter(None, re.sub(r"[^\w\d'\s]+",' ',text).split(' '))))
        glove_feats = []
        for word in clean_string.split(' '):
            try:
                glove_feats.append(self.word2vec[word])
            except:
                try:
                    glove_feats.append(self.word2vec[word.lower()])
                except:
                    continue
            
        if glove_feats==[]:
            return None
        text_feats = np.asarray(glove_feats)
        if text_feats.shape[0]>=self.max_words:
            return text_feats[:self.max_words,:]
        else:
            try:
                padded_feats = np.concatenate((text_feats, np.zeros((self.max_words-text_feats.shape[0], text_feats.shape[1]))), axis=0)
            except:
                padded_feats = np.concatenate((text_feats, np.zeros((self.max_words-len(glove_feats), 300))), axis=0)
            assert padded_feats.shape[0]==self.max_words
            return padded_feats

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        item = self.dataset[idx]
        audio_path = item['audio_path']
        text = item['text']
        x_vector = item['x_vector']
        audio_feats = self._feature_extraction(audio_path)
        text_feats = self._get_glove_features(text)
        label = LABELS[item['label']]
        return torch.Tensor(audio_feats), torch.Tensor(text_feats), torch.Tensor(x_vector), torch.LongTensor([label])
        
        

if __name__=='__main__':
    dataset_file = '/media/newhd/IEMOCAP_dataset/IEMOCAP_x_vector.pickle'
    glove_path = '/media/newhd/IEMOCAP_dataset/glove.840B.300d.pickle'
    traintest = TrainTest(dataset_file, 1)
    train_splits, valid_splits, test_splits = traintest()
    gen = IEMOCAPDataset(train_splits,glove_path)
    for i in range(len(gen)):
        audio_feats, text_feats, x_vector, label = gen.__getitem__(i)
