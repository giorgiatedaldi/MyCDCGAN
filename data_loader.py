"""Giorgia Tedaldi 339642"""

from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode, num_samples):
        """Initialize and preprocess the CelebA dataset."""
        
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.num_samples = num_samples
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == 'test':
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        
        # Process the attribute file 
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        #random.seed(1234)
        random.shuffle(lines)

        # By specifying 'num_samples' it can be decided on how many images to train the model
        for i, line in enumerate(lines):
            if(self.num_samples and i >= self.num_samples):
                break

            # Splitting line to get the file's name and a list 'values' with its attributes
            split = line.split()
            filename = split[0]
            values = split[1:]

            # Create a custom list 'label' which contains only the desired attributes
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            
            #For the current model the test set is not needed. 
            #It may be useful in future implementations.
            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')
        print(self.__len__) 

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'test':
            dataset = self.test_dataset
        filename, label = dataset[index]

        # =================================================================================== #
        #                    If 'images' folder is divided into subfolder                       #
        # =================================================================================== #
        #num_image = [int(s) for s in re.findall(r'\b\d+\b', filename)][0]
        #num_sub_dir = (num_image // 20000)
        #if (num_image % 10000 != 0):
            #num_sub_dir += 1
        #sub_dir = str(num_sub_dir).zfill(2)
        #print(num_image, result_1)
        #image = Image.open(os.path.join(self.image_dir, sub_dir, filename))

        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, mode='train', num_workers=1, num_samples=None):
    """Build and return a data loader."""
    
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode, num_samples)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

