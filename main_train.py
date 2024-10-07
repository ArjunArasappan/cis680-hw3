import torch
import torch.nn as nn
import torch.optim as optim
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------------------
    # Paths to Data
    # ----------------------------
    imgs_path = '/content/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = '/content/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = "/content/hw3_mycocodata_labels_comp_zlib.npy"
    bboxes_path = "/content/hw3_mycocodata_bboxes_comp_zlib.npy"
    paths = [imgs_path, masks_path, labels_path, bboxes_path]

    # ----------------------------
    # Dataset and DataLoader
    # ----------------------------
    # Data augmentation
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        # TODO: add more if training doesn't train to acceptable accuracy
    ])

    dataset = BuildDataset(paths)

    # train test split
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # dataloaders init
    batch_size = 2
    num_workers = 2 
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = test_build_loader.loader()

    # ----------------------------
    # Model Initialization
    # ----------------------------
    # backbone and SOLO head
    resnet50_fpn = Resnet50Backbone().to(device)
    solo_head = SOLOHead(num_classes=4).to(device)  # num_classes includes background

    # ----------------------------
    # Optimizer and Learning Rate Scheduler
    # ----------------------------
    # learning rate according to batch size
    initial_lr = 0.01 / 8  # Since batch size is 2, 0.01 / (16 / batch_size)
    params = list(solo_head.parameters()) + list(resnet50_fpn.parameters())
    optimizer = optim.SGD(params, lr=initial_lr, momentum=0.9, weight_decay=0.0001)

    # LR scheduler: divide LR by 10 at epochs 27 and 33
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1)

    # ----------------------------
    # Checkpoint Setup
    # ----------------------------
    num_epochs = 36
    start_epoch = 0
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Optionally resume from a checkpoint
    # checkpoint_path = './checkpoints/checkpoint_epoch_XX.pth'
    # if os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     resnet50_fpn.load_state_dict(checkpoint['resnet50_fpn_state_dict'])
    #     solo_head.load_state_dict(checkpoint['solo_head_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Resuming training from epoch {start_epoch}")

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(start_epoch, num_epochs):
        resnet50_fpn.train()
        solo_head.train()
        total_loss_epoch = 0.0

        for iter_num, data in enumerate(train_loader, 0):
            images, labels_list, masks_list, bboxes_list = [data[i] for i in range(len(data))]
            images = images.to(device)

            optimizer.zero_grad()

            # forward thru backbone and solo head
            fpn_feat_list = list(resnet50_fpn(images).values())
            cate_pred_list, ins_pred_list = solo_head(fpn_feat_list, eval=False)

            # targets
            ins_gts_list, ins_ind_gts_list, cate_gts_list = solo_head.target(
                ins_pred_list, bboxes_list, labels_list, masks_list)

            cate_loss, mask_loss, total_loss = solo_head.loss(
                cate_pred_list, ins_pred_list, ins_gts_list, ins_ind_gts_list, cate_gts_list)

            total_loss.backward()
            optimizer.step()

            total_loss_epoch += total_loss.item()

            if iter_num % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Iteration [{iter_num}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}, Cate Loss: {cate_loss.item():.4f}, Mask Loss: {mask_loss.item():.4f}')

        # update LR
        lr_scheduler.step()

        average_loss = total_loss_epoch / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] finished, Average Loss: {average_loss:.4f}')

        # ----------------------------
        # Save Checkpoint
        # ----------------------------
        checkpoint = {
            'epoch': epoch,
            'resnet50_fpn_state_dict': resnet50_fpn.state_dict(),
            'solo_head_state_dict': solo_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    print("Training complete.")

if __name__ == '__main__':
    main()
