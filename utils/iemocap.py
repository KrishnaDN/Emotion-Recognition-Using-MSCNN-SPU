import os
import glob
import kaldi_io
import pickle


def extract_xvectors(audio_root, kaldi_root,store_path):
    audio_files = glob.glob(audio_root+'/*.wav')
    cur_dir = os.getcwd()
    os.chdir(kaldi_root)
    os.makedirs('data/test', exist_ok=True)
    with open('data/test/wav.scp','w') as f_wav, open('data/test/utt2spk','w') as f_u2s, open('data/test/spk2utt','w') as f_s2u:
        for filepath in audio_files:
            f_wav.write(filepath.split('/')[-1]+' '+filepath+'\n')
            f_u2s.write(filepath.split('/')[-1]+' '+filepath.split('/')[-1]+'\n')
            f_s2u.write(filepath.split('/')[-1]+' '+filepath.split('/')[-1]+'\n')
    os.system('sh extract_feats.sh')
    x_vector_ark = '/home/krishna/Krishna/kaldi/egs/news_diarization/v2/exp/xvector_nnet_1a/test_krishna/xvector.1.ark'
    x_vector_dict = dict()
    for key, vec in  kaldi_io.read_vec_flt_ark(x_vector_ark):
        filename = key.split('-')[0]
        x_vector_dict[filename] = vec
    save_dict = dict()
    for filepath in audio_files:
        datum = {}
        datum['audio_path'] = filepath
        datum['text'] = [line.rstrip('\n') for line in open(filepath[:-4]+'.txt')][0].split('\t')[1].split('___')[0]
        datum['label'] = [line.rstrip('\n') for line in open(filepath[:-4]+'.txt')][0].split('\t')[1].split('___')[1]
        datum['x_vector'] = x_vector_dict[filepath.split('/')[-1]]
        save_dict[filepath.split('/')[-1]] = datum
        
    with open(store_path, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

if __name__=='__main__':
    audio_root = '/media/newhd/IEMOCAP_dataset/IEMOCAP_dataset'
    kaldi_root = '/home/krishna/Krishna/kaldi/egs/news_diarization/v2'
    store_path = '/media/newhd/IEMOCAP_dataset/IEMOCAP_x_vector.pickle'
    extract_xvectors(audio_root, kaldi_root)