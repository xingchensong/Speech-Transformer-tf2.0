import librosa,sys
from random import shuffle
from tqdm import tqdm
import numpy as np
sys.path.append("..")
from speechpy import processing
from speechpy import feature
import scipy.io.wavfile as wav
from utils import  AttrDict
import yaml

configfile = open('config/hparams.yaml')
data_config = AttrDict(yaml.load(configfile))

corpus = 'librispeech'
dev_parts = [corpus+'/data/dev-clean.scp', corpus+'/data/dev-other.scp' ]
test_parts = [corpus+'/data/test-clean.scp', corpus+'/data/test-other.scp']
train_parts = [corpus+'/data/train-clean-100.scp',corpus+'/data/train-clean-360.scp', corpus+'/data/train-other-500.scp']

# 32 classes
PAD = 0
UNK = 1
BOS = 2
EOS = 3
PAD_FLAG = '<PAD>'
UNK_FLAG = '<UNK>'
BOS_FLAG = '<BOS>'
EOS_FLAG = '<EOS>'
chars = ['<PAD>','<UNK>','<BOS>','<EOS>','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q',
         'r','s','t','u','v','w','x','y','z','_','\'']

class DataFeeder(object):
    '''
        属性：
            wav_lst :    ['LibriSpeech_small/dev-clean/84/121550/84-121550-0000.wav' , ...]
            pny_lst :    [['g','o','o','d','_'] , ...]
            vocab :   ['_', 'y', ...]
        '''

    def __init__(self, config, Train_or_Test):
        self.data_type = Train_or_Test  # train test dev
        self.data_path = config.data_path  # 存放数据的顶层目录
        self.data_length = config.data_length  # 使用多少数据训练 None表示全部
        self.batch_size = config.__getattr__(Train_or_Test).batch_size  # batch大小
        self.shuffle = config.shuffle  # 是否打乱训练数据
        self.feature_type = config.feature_type # fbanks
        self.frame_rate = config.frame_rate
        self.apply_cmvn = config.apply_cmvn
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length

        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            for i in train_parts:
                read_files.append(i)
        elif self.data_type == 'dev':
            for i in dev_parts:
                read_files.append(i)
        elif self.data_type == 'test':
            for i in test_parts:
                read_files.append(i)

        self.wav_lst = []
        self.char_list = []
        for file in read_files:
            print('load ', file, ' data...')
            with open(file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, text, id = line.split('\t')
                self.wav_lst.append(wav_file)
                self.char_list.append(list(text))

        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.char_list = self.char_list[:self.data_length]

        self.lengths = len(self.wav_lst)
        print('make vocab...')
        self.vocab = self.mk_vocab()

    def char2id(self,line, vocab):
        ids = []
        for char in line:
            if char in vocab:
                ids.append(vocab.index(char))
            else:
                ids.append(vocab.index('<UNK>'))
        return ids

    def wav_padding(self, wav_data_lst):
        feature_dim = np.array(wav_data_lst).shape[-1] # 80 mels
        wav_lens = np.array([len(data) for data in wav_data_lst])  # len(data)实际上就是求语谱图的第一维的长度，也就是n_frames
        # 取一个batch中的最长
        wav_max_len = max(wav_lens)
        # TODO: 1-D conv是 三维 ，2-D conv是四维
        new_wav_data_lst = np.zeros(
            (len(wav_data_lst), wav_max_len, feature_dim, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        # print('new_wav_data_lst',new_wav_data_lst.shape,wav_lens.shape)
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_vocab(self):
        vocab = chars
        return vocab

    def get_batch(self):
        # shuffle只是对index打乱，没有对原始数据打乱，所以wav_lst[i]和pny_lst[i]还是一一对应的
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            # len(self.wav_lst) // self.batch_size的值 表示一个epoch里有多少step才能把所有数据过一遍
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []  # wav_data_lst里放的是batch_size个频谱图 wav_lst里放的是音频文件地址
                label_data_lst = []
                ground_truth_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:
                    # TODO：计算频谱图
                    fbank, n_frames = compute_fbank(self.data_path +'/'+ self.wav_lst[index])

                    pad_fbank = fbank
                    label = self.char2id( [BOS]+self.char_list[index], self.vocab)
                    g_truth = self.char2id( self.char_list[index]+[EOS], self.vocab)
                    wav_data_lst.append(pad_fbank)
                    label_data_lst.append(label)
                    ground_truth_lst.append(g_truth)

                # TODO：对语谱图时间维度进行第二次pad，pad成本次batch中最长的长度
                # TODO: label是decoder输入 g_truth是decoder的target
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)
                pad_label_data, label_length = self.label_padding(label_data_lst)
                pad_ground_truth, _ = self.label_padding(ground_truth_lst)

                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length.reshape(-1, 1),
                          'label_length': label_length.reshape(-1, 1),  # batch中的每个utt的真实长度
                          'ground_truth': pad_ground_truth
                          }
                print('genarate one batch mel data')
                yield inputs
        pass

    def __len__(self):
        return self.lengths




def compute_fbank(file):
    sr, signal = wav.read(file)
    signal = signal[:, 0]
    # Pre-emphasizing.
    signal_preemphasized = processing.preemphasis(signal, cof=data_config.preemphasis)
    # Stacking frames
    frames = processing.stack_frames(signal, sampling_frequency=sr,
                                     frame_length=data_config.window_size,
                                     frame_stride=data_config.hop_size,
                                     zero_padding=True)

    # Extracting power spectrum
    power_spectrum = processing.power_spectrum(frames, fft_points=512) # num_frames x fft_length
    print('power spectrum shape=', power_spectrum.shape)

    ############# Extract fbanks features #############
    log_fbank = feature.lmfe(signal,sampling_frequency=sr,frame_length=data_config.window_size,
                        frame_stride=data_config.hop_size,num_filters=data_config.num_mels,
                         fft_length=512, low_frequency=0,high_frequency=None) # num_frames x num_filters


    if data_config.apply_cmvn:
        # Cepstral mean variance normalization.
        log_fbank_cmvn = processing.cmvn(log_fbank, variance_normalization=True)
        print('fbank(mean + variance normalized) feature shape=', log_fbank_cmvn.shape)
        log_fbank = log_fbank_cmvn

    # Extracting derivative features
    log_fbank_feature_cube = feature.extract_derivative_feature(log_fbank)
    print('log fbank feature cube shape=', log_fbank_feature_cube.shape) # num_frames x num_filters x 3

    return log_fbank_feature_cube

if __name__=='__main__':
    a = 'Do you '
    print(list(a.lower().replace(' ','_')))