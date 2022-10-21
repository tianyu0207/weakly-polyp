import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import os

class Dataset(data.Dataset):
    def __init__(self, args,  is_normal=True, test_mode=False, sampling='random', transform = None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.num_segments = 32
        self.sampling = sampling
     
        self.test_mode = test_mode

        self.normal_root_path_train = "/media/yuanhong/18TB/Yu/Yu_HDD/colon_video_features/train_set/colon_i3d_feature_train_normal"
        self.abnormal_root_path_train = "/media/yuanhong/18TB/Yu/Yu_HDD/colon_video_features/train_set/colon_i3d_feature_train_abnormal"

        self.normal_root_path_test = "/media/yuanhong/18TB/Yu/Yu_HDD/colon_video_features/test_set/colon_i3d_feature_test_normal"
        self.abnormal_root_path_test = "/media/yuanhong/18TB/Yu/Yu_HDD/colon_video_features/test_set/colon_i3d_feature_test_abnormal"
      
        self.tranform = transform
        self.list = self._parse_list()
        self.num_frame = 0
        self.labels = None
        

    def _parse_list(self):

        normal_file_list = sorted(os.listdir(self.normal_root_path_train))
        abnormal_file_list = sorted(os.listdir(self.abnormal_root_path_train))

        normal_file_list_test = sorted(os.listdir(self.normal_root_path_test))
        abnormal_file_list_test = sorted(os.listdir(self.abnormal_root_path_test))
        if self.test_mode is False:
          if self.is_normal:
              l = [self.normal_root_path_train + '/' + s  for s in normal_file_list]

            #   print('normal list for colon')
            #   print(l)
          else:
              l = [self.abnormal_root_path_train + '/' + s  for s in abnormal_file_list]
            #   print('abnormal list for colon')
            #   print(l)
        else:
         
          l = [self.normal_root_path_test + '/' + s  for s in normal_file_list_test] + [self.abnormal_root_path_test + '/' + s  for s in abnormal_file_list_test]
        return l


    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        file_name = self.list[index].strip('\n')
        features = np.load(file_name, allow_pickle=True)
        features = np.array(features, dtype=np.float32)


        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
          return features, file_name

        features = features.transpose(1, 0, 2)  # [10, B, T, F]

        features = process_feat(features.squeeze(0), 32)  # divide a video into 32 segments
        features = np.array(features, dtype=np.float32)
        features = np.expand_dims(features, 1)

        # data, vid_num_seg, sample_idx = self.get_data(index)
        # temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        # return features, label, file_name

        return features, label

    def get_data(self, index):
        # vid_name = self.vid_list[index]

        vid_num_seg = 0
        
        feature = np.load(self.list[index].strip('\n'), allow_pickle=True)
        
        vid_num_seg = feature.shape[0]
        
        if self.sampling == 'random':
            sample_idx = self.random_perturb(feature.shape[0])
        elif self.sampling == 'uniform':
            sample_idx = self.uniform_sampling(feature.shape[0])
        else:
            raise AssertionError('Not supported sampling !')
        
        feature = feature[sample_idx]
        
        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    # def get_label(self, index, vid_num_seg, sample_idx):
    def get_label(self):
        
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label


    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
