import os
import torch.nn as nn



class IEMOCAPDataset:
    def __init__(self, dataset_root, glove_path, max_frames, max_words):
        self.dataset_root = dataset_root
        self.glove_path = glove_path
        self.max_frames = max_frames
        self.max_words = max_words
        
    def feature_extraction(self,):
        pass
    
    def _load_glove(self,):
        pass
    
    def __len__(self):
        return len(self.json_links)
        
    
    def __getitem__(self, idx):
        json_link =self.json_links[idx]
        masked_features,original_feats,final_phn_seq,phn_seq_len = utility.load_data(json_link)
        #lang_label=lang_id[self.audio_links[idx].split('/')[-2]]
        sample = {'masked_feats': torch.from_numpy(np.ascontiguousarray(masked_features)), 
                  'gt_feats': torch.from_numpy(np.ascontiguousarray(original_feats)),
                  'phn_seq': torch.from_numpy(np.ascontiguousarray(final_phn_seq)),
                  'labels_length': torch.from_numpy(np.ascontiguousarray(phn_seq_len))}
        return sample
    
    
    
    
    
    