"""
Dataset parser
"""
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, stems=False, transform=None):
        self.type = type_
        self.root_dir = root_dir
        self.stems = stems
        # get images 
        image_list = glob.glob(os.path.join(self.root_dir, 'images/{}'.format(self.type), '*.png'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        if self.type != 'test':
          # get global and part annotation
          global_instance_list = glob.glob(os.path.join(self.root_dir, 'annos/{}/global'.format(self.type), '*.semantic'))
          parts_instance_list = glob.glob(os.path.join(self.root_dir, 'annos/{}/parts'.format(self.type), '*.semantic'))
          global_instance_list.sort()
          parts_instance_list.sort()
          self.global_instance_list = global_instance_list
          self.parts_instance_list = parts_instance_list
          print("# global instance files: ", len(self.global_instance_list))
          print("# part instance files: ", len(self.parts_instance_list))
   
          # check if there are additional annotations for the stem location
          self.stem_list = []
          path_to_stem_anno = os.path.join(self.root_dir, 'annos', self.type, 'stems')
          if stems:
              stem_list = glob.glob(os.path.join(path_to_stem_anno, '*.npy'))
              stem_list.sort()
              self.stem_list = stem_list
              print("# stem anno files: ", len(self.stem_list))


        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        print('MyDataset created - [{} file(s)]'.format(self.real_size))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image
        image = Image.open(self.image_list[index])
        width, height = image.size
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type != 'test':
          # load labels and instances
          global_annos = np.fromfile(self.global_instance_list[index], dtype=np.uint32)
          global_annos = global_annos.reshape(height, width)
   
          parts_annos = np.fromfile(self.parts_instance_list[index], dtype=np.uint32)
          parts_annos = parts_annos.reshape(height, width)
   
          global_labels = global_annos & 0xffff # get lower 16-bits
          global_instances = global_annos >> 16 # get upper 16-bits
          # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
          global_instance_ids = np.unique(global_instances)[1:] # no background
          global_instances_successive =  np.zeros_like(global_instances)
          for idx, id_ in enumerate(global_instance_ids):
              instance_mask = global_instances == id_
              global_instances_successive[instance_mask] = idx + 1
          global_instances = global_instances_successive
   
          assert np.max(global_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'
   
          parts_labels = parts_annos & 0xffff # get lower 16-bits
          parts_instances = parts_annos >> 16 # get upper 16-bits
          # instance ids might start at high numbers and are not successive, thus we make sure that this is the case
          parts_instance_ids = np.unique(parts_instances)[1:] # no background
          parts_instances_successive =  np.zeros_like(parts_instances)
          for idx, id_ in enumerate(parts_instance_ids):
              instance_mask = parts_instances == id_
              parts_instances_successive[instance_mask] = idx + 1
          parts_instances = parts_instances_successive
   
          assert np.max(parts_instances) <= 255, 'Currently we do not suppot more than 255 instances in an image'
   
          global_labels = Image.fromarray(np.uint8(global_labels))
          # TODO there might be more than 255 instances
          global_instances = Image.fromarray(np.uint8(global_instances))
          
          parts_labels = Image.fromarray(np.uint8(parts_labels))
          # TODO there might be more than 255 instances
          parts_instances = Image.fromarray(np.uint8(parts_instances))
   
          sample['global_instances'] = global_instances
          sample['global_labels'] = global_labels
          
          sample['parts_instances'] = parts_instances
          sample['parts_labels'] = parts_labels
   
          # load stems if provided
          if self.stems:
              stem_anno = np.fromfile(self.stem_list[index], dtype=np.uint8)
              stem_anno = stem_anno.reshape(height, width)
              stem_anno = Image.fromarray(np.uint8(stem_anno))
              
              sample['stem_anno'] = stem_anno

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)

        return sample
