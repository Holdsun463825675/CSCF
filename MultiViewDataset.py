import scipy.io as io
import numpy as np
from torch.utils.data import Dataset

def GetMultiDataset(data_folder='../data', dataset='ActivityNet', text_features='clip', type='same', mode='multi', train=True):
    if text_features == 'clip':
        text_feats = 'text_att'
        similarity = 'similarity_text'
    elif text_features == 'clipExtend':
        text_feats = 'extend_text_att'
        similarity = 'similarity_extend_text'
    else:
        raise ValueError(f'Not supported {text_features}.')

    data_dir = f'{data_folder}/{dataset}'
    sim_mat_root = data_dir + '/similarity_matrix.mat'
    if train:
        root1 = data_dir + f'/{dataset}1/train_data.mat'
        root2 = data_dir + f'/{dataset}2/train_data.mat'
    else:
        root1 = data_dir + f'/{dataset}1/test_data.mat'
        root2 = data_dir + f'/{dataset}2/test_data.mat'


    data1 = io.loadmat(root1)
    data2 = io.loadmat(root2)
    sim_mat = io.loadmat(sim_mat_root)[similarity].astype(np.float32)

    audio_data1 = data1['audio_att'].astype(np.float32)
    video_data1 = data1['video_att'].astype(np.float32)
    num_class_data1 = data1['text_att'].astype(np.float32).shape[0]
    label1 = data1['label'].squeeze() - 1
    audio_data2 = data2['audio_att'].astype(np.float32)
    video_data2 = data2['video_att'].astype(np.float32)
    num_class_data2 = data2['text_att'].astype(np.float32).shape[0]
    label2 = data2['label'].squeeze() - 1
    label2 = label2 + num_class_data1
    text_att1 = data1[text_feats].astype(np.float32)
    text_att2 = data2[text_feats].astype(np.float32)

    masks1 = np.zeros(num_class_data1)
    masks2 = np.ones(num_class_data2)
    masks = np.concatenate((masks1, masks2), axis=0)
    audio_seen_masks = (masks == 0)
    video_seen_masks = (masks == 1)
    seen_masks = [audio_seen_masks, video_seen_masks]

    len_data1 = len(label1)
    len_data2 = len(label2)
    audio_dimension = audio_data1.shape[1]
    video_dimension = video_data1.shape[1]
    audio_zero1 = np.zeros((len_data1, audio_dimension)).astype(np.float32)
    video_zero1 = np.zeros((len_data1, video_dimension)).astype(np.float32)
    audio_zero2 = np.zeros((len_data2, audio_dimension)).astype(np.float32)
    video_zero2 = np.zeros((len_data2, video_dimension)).astype(np.float32)

    if mode == 'multi':
        if type == "same":
            audio = np.concatenate((audio_data1, audio_zero2), axis=0)
            video = np.concatenate((video_zero1, video_data2), axis=0)
            label = np.concatenate((label1, label2), axis=0)
            mask = np.concatenate((np.full(len_data1, 0, dtype=int), np.full(len_data2, 1, dtype=int)), axis=0)
        elif type == "contra":
            audio = np.concatenate((audio_zero1, audio_data2), axis=0)
            video = np.concatenate((video_data1, video_zero2), axis=0)
            label = np.concatenate((label1, label2), axis=0)
            mask = np.concatenate((np.full(len_data1, 1, dtype=int), np.full(len_data2, 0, dtype=int)), axis=0)
        elif type == "data1":
            audio = audio_data1
            video = video_data1
            label = label1
            mask = np.full(len_data1, 2, dtype=int)
        elif type == "data2":
            audio = audio_data2
            video = video_data2
            label = label2
            mask = np.full(len_data2, 2, dtype=int)
        elif type == "total":
            audio = np.concatenate((audio_data1, audio_data2), axis=0)
            video = np.concatenate((video_data1, video_data2), axis=0)
            label = np.concatenate((label1, label2), axis=0)
            mask = np.full(len_data1 + len_data2, 2, dtype=int)
        elif type == "mixture":
            audio = np.concatenate((audio_data1, audio_zero2, audio_zero1, audio_data2, audio_data1, audio_data2), axis=0)
            video = np.concatenate((video_zero1, video_data2, video_data1, video_zero2, video_data1, video_data2), axis=0)
            label = np.concatenate((label1, label2, label1, label2, label1, label2), axis=0)
            mask = np.concatenate((np.full(len_data1, 0, dtype=int), np.full(len_data2, 1, dtype=int), np.full(len_data1, 1, dtype=int), np.full(len_data2, 0, dtype=int), np.full(len_data1 + len_data2, 2, dtype=int)), axis=0)
        else:
            raise ValueError(f'Not supported data_type {type}.')
        text = np.concatenate((text_att1, text_att2), axis=0)
        similarity_mat = sim_mat
    elif mode == 'audio' or mode == 'video':
        if type == "same":
            if mode == 'audio':
                audio = audio_data1
                video = video_zero1
                label = label1
                mask = np.full(len_data1, 0, dtype=int)
            else:
                audio = audio_zero2
                video = video_data2
                label = label2
                mask = np.full(len_data2, 1, dtype=int)
        elif type == "contra":
            if mode == 'audio':
                audio = audio_data2
                video = video_zero2
                label = label2
                mask = np.full(len_data2, 0, dtype=int)
            else:
                audio = audio_zero1
                video = video_data1
                label = label1
                mask = np.full(len_data1, 1, dtype=int)
        elif type == "data1":
            if mode == 'audio':
                audio = audio_data1
                video = video_zero1
                mask = np.full(len_data1, 0, dtype=int)
            else:
                audio = audio_zero1
                video = video_data1
                mask = np.full(len_data1, 1, dtype=int)
            label = label1
        elif type == "data2":
            if mode == 'audio':
                audio = audio_data2
                video = video_zero2
                mask = np.full(len_data2, 0, dtype=int)
            else:
                audio = audio_zero2
                video = video_data2
                mask = np.full(len_data2, 1, dtype=int)
            label = label2
        elif type == "total":
            if mode == 'audio':
                audio = np.concatenate((audio_data1, audio_data2), axis=0)
                video = np.concatenate((video_zero1, video_zero2), axis=0)
                mask = np.full(len_data1 + len_data2, 0, dtype=int)
            else:
                audio = np.concatenate((audio_zero1, audio_zero2), axis=0)
                video = np.concatenate((video_data1, video_data2), axis=0)
                mask = np.full(len_data1 + len_data2, 1, dtype=int)
            label = np.concatenate((label1, label2), axis=0)
        else:
            raise ValueError(f'Not supported data_type {type}.')
        text = np.concatenate((text_att1, text_att2), axis=0)
        similarity_mat = sim_mat
    else:
        raise ValueError(f'Not supported data_mode {mode}.')

    indices = np.random.permutation(len(label))
    audio = audio[indices]
    video = video[indices]
    label = label[indices]
    mask = mask[indices]

    imputs_multi = [audio, video, label, mask]
    return imputs_multi, text, similarity_mat, seen_masks

class MultiViewDataset(Dataset):
    def __init__(self, data_folder='../data', dataset='ActivityNet', text_features='clip', type='same', mode='multi', train=True):
        self.inputs_multi, self.attrs_multi, self.similarity_mat, self.seen_masks = GetMultiDataset(data_folder, dataset, text_features, type, mode, train)

    def __len__(self):
        return len(self.inputs_multi[0])

    def __getitem__(self, idx):
        audio = self.inputs_multi[0][idx]
        video = self.inputs_multi[1][idx]
        label = self.inputs_multi[2][idx]
        mask = self.inputs_multi[3][idx]
        return audio, video, label, mask

def custom_collate_fn(batch):
    audio = [item[0] for item in batch]
    video = [item[1] for item in batch]
    audio = np.array(audio)
    video = np.array(video)
    inputs_multi = [audio, video]
    label = [item[2] for item in batch]
    mask = [item[3] for item in batch]
    return inputs_multi, label, mask
