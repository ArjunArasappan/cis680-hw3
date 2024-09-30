## Author: Lishuo Pan 2020/4/18

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        # TODO: load dataset, make mask list

        img_path, mask_path, label_path , bbox_path= paths

        # print("Current working directory:", os.getcwd())
        # print("Files in data directory:", os.listdir(data_root))

        # load shit
        with h5py.File(img_path, 'r') as file:
            self.images = file[list(file.keys())[0]][:]

        with h5py.File(mask_path, 'r') as file:
            self.masks = file[list(file.keys())[0]][:]
            
            
        self.bboxes = np.load(bbox_path, allow_pickle=True)
        self.labels = np.load(label_path, allow_pickle=True)
        
        print(len(self.masks), len(self.images), len(self.bboxes), len(self.labels))
        print(self.masks.shape, self.images.shape, self.bboxes[0].shape, self.labels[0].shape)
        print(self.labels[0])

        
        



    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        
    def __getitem__(self, index):
        # TODO: __getitem__

        # check flag
        # print(self.images.shape)
        img = self.images[index]
        mask = self.masks[index]
        bbox = self.bboxes[index]

    
        tensor = torch.tensor(img)
        
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        tensor = F.interpolate(tensor.unsqueeze(0), size=(800, 1066), mode='bilinear', align_corners=False).squeeze(0)

        tensor = transforms.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # print(tensor.shape)
 
        
        tensor = F.pad(tensor, (11, 11, 0, 0), "constant", 0)

        

        # print(list(tensor.shape))
        assert list(tensor.shape) == [3, 800, 1088]
        
        print(self.bboxes[0].shape, self.masks[0].shape)
        assert self.bboxes[index].shape[0] == self.masks[index].shape[0]
        
        return tensor, self.labels[index], self.masks[index], self.bboxes[index]
    
    
    def __len__(self):
        return len(self.images)
    

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        # TODO: image preprocess

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]
        return img, mask, bbox


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Initialize the DataLoader with the custom collect function
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=self.num_workers, collate_fn=self.collect_fn)


    # output:
        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
        # img: (bz, 3, 300, 400)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
            
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list


    def loader(self):
        return self.dataloader

## Visualize debugging
if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    ## Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]


        # plot the origin img
        for i in range(batch_size):
            ## TODO: plot images with annotations
            plt.savefig("./testfig/visualtrainset"+str(iter)+".png")
            plt.show()

        if iter == 10:
            break

