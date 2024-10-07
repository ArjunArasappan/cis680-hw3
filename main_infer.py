# main_infer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from dataset import BuildDataset, BuildDataLoader
from backbone import Resnet50Backbone
from solo_head import SOLOHead

def main():
    # ----------------------------
    # Device Configuration
    # ----------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # Paths to Data and Checkpoint
    # ----------------------------
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    checkpoint_path = './checkpoints/checkpoint_epoch_36.pth'  # Replace with your checkpoint path
    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    # ----------------------------
    # Dataset and DataLoader
    # ----------------------------
    # Load the dataset
    dataset = BuildDataset(paths)

    # Split dataset into training and testing sets
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader parameters
    batch_size = 1  # Set batch size to 1 for inference
    num_workers = 2  # Adjust based on your system

    # Build DataLoader for test set
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = test_build_loader.loader()

    # ----------------------------
    # Model Initialization
    # ----------------------------
    # Initialize backbone and SOLO head
    resnet50_fpn = Resnet50Backbone().to(device)
    solo_head = SOLOHead(num_classes=4).to(device)  # num_classes includes background

    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        resnet50_fpn.load_state_dict(checkpoint['resnet50_fpn_state_dict'])
        solo_head.load_state_dict(checkpoint['solo_head_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Set models to evaluation mode
    resnet50_fpn.eval()
    solo_head.eval()

    # ----------------------------
    # Inference Loop
    # ----------------------------
    color_list = ["jet", "ocean", "Spectral"]  # For visualization

    # Directory for saving inference images
    os.makedirs('./infer_imgs', exist_ok=True)

    with torch.no_grad():
        for iter_num, data in enumerate(test_loader, 0):
            images, labels_list, masks_list, bboxes_list = [data[i] for i in range(len(data))]
            images = images.to(device)

            # Forward pass through backbone
            fpn_feat_list = list(resnet50_fpn(images).values())

            # Forward pass through SOLO head in eval mode
            cate_pred_list, ins_pred_list = solo_head(fpn_feat_list, eval=True)

            print(f"dimensions of cate pred and ins pred: {cate_pred_list[0].shape}, {ins_pred_list[0].shape}")

            # Original image size (before any resizing or padding)
            ori_size = [images.shape[2], images.shape[3]]  # (ori_H, ori_W)

            # Post-process predictions
            NMS_sorted_scores_list, NMS_sorted_cate_label_list, NMS_sorted_ins_list = solo_head.PostProcess(
                ins_pred_list, cate_pred_list, ori_size)

            # Visualize and save results
            solo_head.PlotInfer(NMS_sorted_scores_list,
                                NMS_sorted_cate_label_list,
                                NMS_sorted_ins_list,
                                color_list,
                                images,
                                iter_num)

            print(f"Inference done for batch {iter_num}")

    print("Inference complete.")

if __name__ == '__main__':
    main()
