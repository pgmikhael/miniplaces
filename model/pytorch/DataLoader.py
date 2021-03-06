import os
import numpy as np
import scipy.misc
import h5py
import torch
from PIL import Image
from torchvision import transforms

np.random.seed(123)


# transforms



# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])
        self.lab_set = np.array(f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        self.shuffle()
        self._idx = 0
        
    def next_batch(self, batch_size):
        labels_batch = np.zeros(batch_size)
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        
        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]
            images_batch[i, ...] = image
            labels_batch[i, ...] = self.lab_set[self._idx]

            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.randomize:
                    self.shuffle()
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        # self.fine_size = int(kwargs['fine_size'])
        self.data_mean = kwargs['data_mean']
        self.data_sd = kwargs['data_sd']
        self.transform = kwargs['transform']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f: #open the file to 'r' read
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        # self.list_lab = np.array(self.list_lab, np.int64)
        self.list_lab = torch.LongTensor(self.list_lab)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = torch.zeros([batch_size, 3, self.load_size, self.load_size]) 
        labels_batch = torch.zeros(batch_size,  dtype=torch.long)
        for i in range(batch_size):
            image = Image.open(self.list_im[self._idx])
            # image = scipy.misc.imread(self.list_im[self._idx])
            # image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            # image = image.astype(np.float32)/255.
            # image = image - self.data_mean


            # transform into PIL image
            data_transform = transforms.Compose([
                # transforms.RandomResizedCrop(self.load_size),
                transforms.Resize(self.load_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                        mean=self.data_mean,
                        std=self.data_sd)
                ])
            
            no_transform = transforms.Compose([
                    transforms.Resize(self.load_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=self.data_mean,
                        std=self.data_sd)])

            if self.transform:
                tsfm = np.random.random_integers(0, 1)
                if tsfm > 0:

                    images_batch[i, ...] = data_transform(image)
                else:
                    images_batch[i, ...] = no_transform(image)

            # if self.randomize:
            #     flip = np.random.random_integers(0, 1)
            #     if flip>0:
            #         image = image[:,::-1,:]
            #     offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
            #     offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            # else:
            #     offset_h = (self.load_size-self.fine_size)//2
            #     offset_w = (self.load_size-self.fine_size)//2

            # images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            # labels_batch[i, ...] = self.list_lab[self._idx]
            # images_batch[i, ...] = image
            
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
    



        # transform into pytorch tensors
        # images_batch = torch.from_numpy(images_batch)
        # images_batch = images_batch.view(batch_size, 3, self.load_size, self.load_size)
        # labels_batch = torch.from_numpy(labels_batch)

        return images_batch, labels_batch
        #to(torch.int64)
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

class TestLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root']) #'../../data2/images/test'
        for r, d, files in os.walk(self.data_root):  
            self.filenames = files
        self.filenames.sort()
        # for each element in file name populate a tensor with that image in order
        self.list_im = []
        for file in self.filenames:
            self.list_im.append(os.path.join(self.data_root, file))
        self.list_im = np.array(self.list_im, np.object)

        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)


        #permutation
        #perm = np.random.permutation(self.num) 
        #self.list_im[:, ...] = self.list_im[perm, ...]
        #self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.load_size, self.load_size, 3)) 
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            # image = image.astype(np.float32)/255.
            # image = image - self.data_mean
            # if self.randomize:
            #     flip = np.random.random_integers(0, 1)
            #     if flip>0:
            #         image = image[:,::-1,:]
            #     offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
            #     offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            # else:
            #     offset_h = (self.load_size-self.fine_size)//2
            #     offset_w = (self.load_size-self.fine_size)//2

            # images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            # labels_batch[i, ...] = self.list_lab[self._idx]
            images_batch[i, ...] = image
            
            self._idx += 1
            flnames = self.filenames[self._idx-batch_size:self._idx]
            if self._idx == self.num:
                self._idx = 0
        
        # transform into pytorch tensors
        images_batch = torch.from_numpy(images_batch)
        images_batch = images_batch.view(batch_size, 3, self.load_size, self.load_size)

        return images_batch.int(), flnames
        #to(torch.int64)
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

class ValLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root']) #'../../data2/images/test'
        for r, d, files in os.walk(self.data_root):  
            self.filenames = files
        self.filenames.sort()
        # for each element in file name populate a tensor with that image in order
        self.list_im = []
        for file in self.filenames:
            self.list_im.append(os.path.join(self.data_root, file))
        self.list_im = np.array(self.list_im, np.object)

        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)


        #permutation
        #perm = np.random.permutation(self.num) 
        #self.list_im[:, ...] = self.list_im[perm, ...]
        #self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0

    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.load_size, self.load_size, 3)) 
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            images_batch[i, ...] = image
            
            self._idx += 1
            flnames = self.filenames[self._idx-batch_size:self._idx]
            if self._idx == self.num:
                self._idx = 0
        
        # transform into pytorch tensors
        images_batch = torch.from_numpy(images_batch)
        images_batch = images_batch.view(batch_size, 3, self.load_size, self.load_size)

        return images_batch.int(), flnames
        #to(torch.int64)
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
