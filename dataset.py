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
        # Paths to the data
        img_path, mask_path, label_path, bbox_path = paths
        
        # Loading the data
        with h5py.File(img_path, 'r') as file:
            self.images = file[list(file.keys())[0]][:100]
        with h5py.File(mask_path, 'r') as file:
            self.masks = file[list(file.keys())[0]][:300]
        self.labels = np.load(label_path, allow_pickle=True)[:100].astype(np.ndarray) 
        self.bboxes = np.load(bbox_path, allow_pickle=True)[:100].astype(np.ndarray) 
        
        # self.labels = torch.from_numpy(self.labels)
        # self.bboxes = torch.from_numpy(self.bboxes)
        
        #  # Loading the data
        # with h5py.File(img_path, 'r') as file:
        #     self.images = file[list(file.keys())[0]][:30]
        # with h5py.File(mask_path, 'r') as file:
        #     self.masks = file[list(file.keys())[0]][:100]
        # self.labels = np.load(label_path, allow_pickle=True)[:30]
        # self.bboxes = np.load(bbox_path, allow_pickle=True)[:30]
        
        
        self.grouped_masks = np.ndarray( (len(self.images)), dtype=object)
    
        idx = 0
        for i, label in enumerate(self.labels):
            n_obj = len(label)
            masks = self.masks[idx : idx + n_obj]
            idx += n_obj
            masks = torch.tensor(masks)
            self.grouped_masks[i] = masks
            
            self.labels[i] = torch.tensor(label)
            self.bboxes[i] = torch.tensor(self.bboxes[i])
            self.images[i] = torch.tensor(self.images[i])
            

        
        # self.labels = torch.from_numpy(self.labels)
            
        # print(self.grouped_masks.shape, self.images.shape, self.labels.shape, self.bboxes.shape)
        
    def transform_img(self, img, is_img = False, is_bbox = False):
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img)
        elif isinstance(img, torch.Tensor):
            tensor = img.clone().detach()
        else:
            raise TypeError("Input img must be a NumPy array or a torch.Tensor")
        

        
        if is_img:
            tensor = tensor / 255

        tensor = tensor.float()

        # print(tensor.shape)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(800, 1066), mode='bilinear', align_corners=False).squeeze(0)
        # print(tensor.shape)
        
        if is_img:
            tensor = transforms.functional.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        tensor = F.pad(tensor, (11, 11, 0, 0), "constant", 0)
        # print(is_img, tensor.shape)

        return tensor
            

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
        mask = self.grouped_masks[index]
        bbox = self.bboxes[index]
        label = self.labels[index]

    
        transed_img = self.transform_img(img, is_img = True)
        transed_mask = self.transform_img(mask)
        
        # NEED TO ADD 11 HERE?
        for i in range(bbox.shape[0]):
            bbox[i][0] = int(bbox[i][0] * 8 / 3)
            bbox[i][1] = int(bbox[i][1] * 8 / 3) + 11
            bbox[i][2] = int(bbox[i][2] * 8 / 3)
            bbox[i][3] = int(bbox[i][3] * 8 / 3) + 11

        

        # print(list(transed_img.shape))
        assert list(transed_img.shape) == [3, 800, 1088]
        
        assert bbox.shape[0] == transed_mask.shape[0]
        
        return transed_img, label, transed_mask, bbox
    
    
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
        

    
        transed_img = self.transform_img(img, is_img = True)
        transed_mask = self.transform_img(mask)
        
        # NEED TO ADD 11 HERE?
        for i in range(bbox.shape[0]):
            bbox[i][0] = int(bbox[i][0] * 8 / 3)
            bbox[i][1] = int(bbox[i][1] * 8 / 3) + 11
            bbox[i][2] = int(bbox[i][2] * 8 / 3)
            bbox[i][3] = int(bbox[i][3] * 8 / 3) + 11

        


        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert bbox.shape[0] == transed_mask.squeeze(0).shape[0]
        return transed_img, transed_mask, bbox


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
            
            
            
        transed_img_list = torch.stack(transed_img_list, dim=0)
        
        # print(transed_img_list[0].shape, label_list[0].shape, transed_mask_list[0].shape, transed_bbox_list[0].shape)
        
        return transed_img_list, label_list, transed_mask_list, transed_bbox_list


    def loader(self):
        return self.dataloader

def part_a_dataset():
    # Paths and dataset loading
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    dataset = BuildDataset(paths)

    # Dataloader setup
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    batch_size = 2
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    color_map = {
        0: 'red',   
        1: 'blue',  
        2: 'green', 
        3: 'yellow'
    }

    for iter, (imgs, labels, masks, bboxes) in enumerate(train_loader):
        imgs = imgs.to(device)
        for i in range(batch_size):
            fig, ax = plt.subplots(1)
            img = imgs[i].cpu().numpy().transpose(1, 2, 0)
            
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize

            ax.imshow(img)
            
            label_colors = {
                0: [1, 0, 0, 0.5],  
                1: [0, 0, 1, 0.5], 
                2: [0, 1, 0, 0.5], 
            }

            for label, mask in zip(labels[i], masks[i]):
                mask = mask.cpu().numpy()
                color = label_colors.get(label.item(), [1, 1, 1, 0.5])  # Default to white if no color is specified
                colored_mask = np.zeros((*mask.shape, 4))  # Create an RGBA image based on mask shape
                for j in range(3):
                    colored_mask[..., j] = color[j] * mask  # Apply color to RGB channels
                colored_mask[..., 3] = color[3] * mask  # Apply alpha channel
                ax.imshow(colored_mask)

            for bbox in bboxes[i]:
                bbox = bbox.cpu().numpy()
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.axis('off')
            plt.savefig(f"./dataset_imgs/visualtrainset_batch{iter}_img{i}.png")
            # plt.show()
            plt.close()

        if iter == 10:
            break


if __name__ == '__main__':
    part_a_dataset()