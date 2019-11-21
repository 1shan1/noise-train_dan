from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence
import os


def readTxt(txtpath):
    filelist = []
    with open(txtpath, 'r') as f:
        for line in f.readlines():
            filelist.append(line.strip())
    return filelist


class NoisyImageGenerator(Sequence):
    def __init__(self, source_image_txt, target_image_txt, batch_size=32, image_size=24):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        # self.source_image_paths = [p for p in Path(source_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        # self.target_image_paths = [p for p in Path(target_image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_image_paths = readTxt(source_image_txt)
        self.target_image_paths = readTxt(target_image_txt)
        self.target_image_txt = target_image_txt
        #self.source_noise_model = source_noise_model
        #self.target_noise_model = target_noise_model
        self.source_image_num = len(self.source_image_paths)
        self.target_image_num = len(self.target_image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        #self.target_image_dir = target_image_dir

        if self.source_image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(source_image_txt))
        if self.target_image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(source_image_txt))

    def __len__(self):
        return self.source_image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        #target_image_dir = self.target_image_dir
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            source_image_path = random.choice(self.source_image_paths)
            # print(source_image_path)
            label_gt = os.path.basename(os.path.dirname(source_image_path))
            re_item = '.*/' + label_gt + '/.*'
            target_image_list = os.popen("grep %s %s | shuf -n 1" %(re_item, self.target_image_txt)).readlines()
            if len(target_image_list)== 0:
                continue
            target_image_path = target_image_list[0].strip()
            if ":" in target_image_path:
                target_image_path = target_image_path.split(":")[-1]
            # print('target_image_list',target_image_list)
            if not os.path.exists(target_image_path) or not os.path.exists(source_image_path):
                print(source_image_path, target_image_list)
                print("Image NOT exists!")
                continue
            source_image = cv2.imread(source_image_path)
            target_image = cv2.imread(target_image_path)
            source_patch = cv2.resize(source_image,(image_size,image_size))
            target_patch = cv2.resize(target_image,(image_size,image_size))
            #h, w, _ = image.shape

            
            #clean_patch = image[i:i + image_size, j:j + image_size]
            x[sample_id] = source_patch
            y[sample_id] = target_patch

            sample_id += 1

            if sample_id == batch_size:
                return x, y


class ValGenerator(Sequence):
    def __init__(self, source_image_txt, target_image_txt, image_size=16):
        self.test_source_image_paths = readTxt(source_image_txt)
        self.test_target_image_paths = readTxt(target_image_txt)
        self.target_image_txt = target_image_txt
        
        self.test_source_image_num = len(self.test_source_image_paths)
        self.test_target_image_num = len(self.test_target_image_paths)
        self.image_size = image_size
        #self.test_target_dir = test_target_dir
        self.data = []

        if self.test_source_image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(source_image_txt))
        if self.test_target_image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(target_image_txt))

        for test_source_image_path in self.test_source_image_paths:

            #filename = os.path.basename(test_source_image_path)
            label_gt = os.path.basename(os.path.dirname(test_source_image_path))
            re_item = '.*/' + label_gt + '/.*'
            target_image_list = os.popen("grep %s %s | shuf -n 1" %(re_item, self.target_image_txt)).readlines()
            # print('1target_image_list',target_image_list)
            if len(target_image_list) ==0:
                continue
            test_target_image_path = target_image_list[0].strip()
            if ":" in test_target_image_path:
                test_target_image_path = test_target_image_path.split(":")[-1]

            #test_target_image_path = self.test_target_dir+'/'+real_fname
            if not os.path.exists(test_target_image_path):
               continue

            y_target = cv2.imread(test_target_image_path)
          
            x_source = cv2.imread(test_source_image_path)
            x = cv2.resize(x_source,(self.image_size,self.image_size))
            #print('test_target_image_path',test_target_image_path)
            #print('y_target:',y_target.shape)
            y = cv2.resize(y_target,(self.image_size,self.image_size))
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
