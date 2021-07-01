import os
import torch.nn as nn
import pickle
import random
import librosa
import numpy as np


LABELS = {'hap':0,'ang':1, 'exc':0,'sad':2,'neu':3}
class TrainTestValid:
    def __init__(self, dataset_file):
        with open(dataset_file, 'rb') as handle:
            self.data = pickle.load(handle)

    def __call__(self,):
        filenames = list(self.data.keys())
        rand_idx = random.sample(range(len(filenames)), int(len(filenames)*0.2))
        train_splits, test_splits, valid_splits = list(),list(),list()
        for i in range(0, int(len(rand_idx)/2)):
            idx = rand_idx[i]
            valid_splits.append(self.data[filenames[idx]])
        for i in range(int(len(rand_idx)/2), len(rand_idx)):
            idx = rand_idx[i]
            test_splits.append(self.data[filenames[idx]])
        for idx in range(len(filenames)):
            if idx in rand_idx:
                continue
            else:
                train_splits.append(self.data[filenames[idx]])
        return train_splits, valid_splits, test_splits

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
        words = []
        with open(self.glove_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                try:
                    vect = np.array(line[1:]).astype(np.float)
                    self.word2vec[word] = vect
                except:
                    continue
    
    def _get_glove_features(self,text):
        import re
        clean_string = ' '.join(list(filter(None, re.sub(r"[^\w\d'\s]+",' ',text).split(' ')))).lower()
        glove_feats = []
        for word in clean_string.split(' '):
            try:
                glove_feats.append(self.word2vec[word])
            except:
                continue
        text_feats = np.asarray(glove_feats)
        if text_feats.shape[0]>=self.max_words:
            return text_feats[:self.max_words,:]
        else:
            padded_feats = np.concatenate((text_feats, np.zeros((self.max_words-text_feats.shape[0], text_feats.shape[1]))), axis=0)
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
        return audio_feats, text_feats, x_vector, label
        
        
        
    

if __name__=='__main__':
    dataset_file = '/media/newhd/IEMOCAP_dataset/IEMOCAP_x_vector.pickle'
    glove_path = '/media/newhd/IEMOCAP_dataset/glove.42B.300d.txt'
    traintestvalid = TrainTestValid(dataset_file)
    train_splits, valid_splits, test_splits = traintestvalid()
    gen = IEMOCAPDataset(train_splits,glove_path)
    audio_feats, text_feats, x_vector, label = gen.__getitem__(1)
    
    
    
    