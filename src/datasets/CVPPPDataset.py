"""
Dataset parser.
"""
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy.core.fromnumeric import transpose
from PIL import Image
from torch.utils.data import Dataset


class CVPPPDataset(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, stems=False, transform=None):
        self.root_dir = root_dir
        self.type = type_

        # get image, foreground and instance list
        image_list = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_rgb.png'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        fg_list = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_fg.png'))
        fg_list.sort()

        self.fg_list = fg_list
        print("# fg files: ", len(fg_list))

        if self.type != 'test':
          label_list = glob.glob(os.path.join(self.root_dir, 'annos/{}/'.format(self.type), '*_label.png'))
          label_list.sort()
          self.label_list = label_list
          print("# label files: ", len(label_list))

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        self.jitter = transforms.ColorJitter(brightness=0.03,
                                            contrast=0.03,
                                            saturation=0.03,
                                            hue=0.03)


        print('CVPPP Dataset created [{}]'.format(self.type))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size-1)
        sample = {}

        # load image and foreground
        image = Image.open(self.image_list[index]).convert('RGB')
        fg = Image.open(self.fg_list[index])
        
        image = image.resize((512,512), resample=Image.BILINEAR)
        # black_canvas = Image.new("RGB", image.size, 0)
        # fg = fg.resize((512,512), resample=Image.NEAREST).convert('L')
        # image = Image.composite(image, black_canvas, fg) # remove background
        
        width, height = image.size
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type != 'test':
          # convert labels to instance map
          labels = skimage.io.imread(self.label_list[index]) # := instance map
          labels = cv2.resize(labels, (512,512), interpolation=cv2.INTER_NEAREST)
          label_ids = np.unique(labels)[1:] # no background
          
          parts_instances = np.zeros((height, width), dtype=np.uint8)
          parts_labels = np.zeros((height, width), dtype=np.uint8)
          instance_counter = 0
          for label_id in label_ids:
              instance_counter = instance_counter + 1
              mask = (labels == label_id)
   
              parts_instances[mask] = instance_counter
              parts_labels[mask] = 1

          # there is only one plant in each image ...
          global_instances = (parts_instances > 0).astype(np.uint8)
          global_labels = parts_labels.copy()

        # --- data augmentation ---
        if self.type == 'train':
          # random hflip
          if random.random() > 0.5:
            # FLIP_TOP_BOTTOM
            sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
            global_instances = np.flip(global_instances, axis=0)
            global_labels = np.flip(global_labels, axis=0)
            parts_instances = np.flip(parts_instances, axis=0)
            parts_labels = np.flip(parts_labels, axis=0)
          
          # random vflip
          if random.random() > 0.5:
            # FLIP_LEFT_RIGHT
            sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
            global_instances = np.flip(global_instances, axis=1)
            global_labels = np.flip(global_labels, axis=1)
            parts_instances = np.flip(parts_instances, axis=1)
            parts_labels = np.flip(parts_labels, axis=1)
  
          # random jittering
          # if random.random() > 0.5:
          #   # need to applied on PIL Image
          #   sample['image'] = self.jitter(sample['image'])
        if self.type != 'test':
          global_instances = Image.fromarray(np.uint8(global_instances))
          global_labels = Image.fromarray(np.uint8(global_labels))
          parts_labels = Image.fromarray(parts_labels)
          parts_instances = Image.fromarray(parts_instances)
   
          sample['global_instances'] = global_instances
          sample['global_labels'] = global_labels
          sample['parts_instances'] = parts_instances
          sample['parts_labels'] = parts_labels

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)

        return sample
