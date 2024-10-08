## Overview

Dataset image examples in [dataset_imgs](dataset_imgs).

FPN image examples in [fpn_imgs](fpn_imgs).
This difference between an object different levels of the FPN feature pyramid is different resolutions, since the levels separate objects by size (in overlapping intervals), as obtained by different-stride convolutions at each level. The lower levels have higher resolution lower semantic info, higher levels are the opposite. 


## Testing scripts
test `solo_head.py` and `dataset.py` by running the `main.py` script. 

colab training: https://colab.research.google.com/drive/11hfi9txNYtI5g8juv4Ef7zOIP82FJ3MT?usp=sharing

can just run main_infer.py in vscode


## First Training run output:
Using device: cuda
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained_backbone' is deprecated since 0.13 and may be removed in the future, please use 'weights_backbone' instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights_backbone' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights_backbone=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights_backbone=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Epoch [1/36], Iteration [0/40], Loss: 3.2687, Cate Loss: 0.8426, Mask Loss: 0.8087
Epoch [1/36], Iteration [10/40], Loss: 2.8755, Cate Loss: 0.4081, Mask Loss: 0.8225
Epoch [1/36], Iteration [20/40], Loss: 2.7636, Cate Loss: 0.3955, Mask Loss: 0.7894
Epoch [1/36], Iteration [30/40], Loss: 2.9107, Cate Loss: 0.3934, Mask Loss: 0.8391
Epoch [1/36] finished, Average Loss: 2.5218
Checkpoint saved at ./checkpoints/checkpoint_epoch_1.pth
Epoch [2/36], Iteration [0/40], Loss: 2.3745, Cate Loss: 0.3922, Mask Loss: 0.6608
Epoch [2/36], Iteration [10/40], Loss: 3.0516, Cate Loss: 0.3916, Mask Loss: 0.8867
Epoch [2/36], Iteration [20/40], Loss: 2.3316, Cate Loss: 0.3911, Mask Loss: 0.6468
Epoch [2/36], Iteration [30/40], Loss: 2.0686, Cate Loss: 0.3911, Mask Loss: 0.5592
Epoch [2/36] finished, Average Loss: 2.3189
Checkpoint saved at ./checkpoints/checkpoint_epoch_2.pth
Epoch [3/36], Iteration [0/40], Loss: 2.4786, Cate Loss: 0.3909, Mask Loss: 0.6959
Epoch [3/36], Iteration [10/40], Loss: 2.6962, Cate Loss: 0.3909, Mask Loss: 0.7684
Epoch [3/36], Iteration [20/40], Loss: 1.7800, Cate Loss: 0.3910, Mask Loss: 0.4630
Epoch [3/36], Iteration [30/40], Loss: 2.5485, Cate Loss: 0.3909, Mask Loss: 0.7192
Epoch [3/36] finished, Average Loss: 2.1695
Checkpoint saved at ./checkpoints/checkpoint_epoch_3.pth
Epoch [4/36], Iteration [0/40], Loss: 2.0832, Cate Loss: 0.3902, Mask Loss: 0.5643
Epoch [4/36], Iteration [10/40], Loss: 0.9162, Cate Loss: 0.3905, Mask Loss: 0.1752
Epoch [4/36], Iteration [20/40], Loss: 1.9437, Cate Loss: 0.3906, Mask Loss: 0.5177
Epoch [4/36], Iteration [30/40], Loss: 2.3311, Cate Loss: 0.3903, Mask Loss: 0.6469
Epoch [4/36] finished, Average Loss: 2.0314
Checkpoint saved at ./checkpoints/checkpoint_epoch_4.pth
Epoch [5/36], Iteration [0/40], Loss: 1.1956, Cate Loss: 0.3906, Mask Loss: 0.2683
Epoch [5/36], Iteration [10/40], Loss: 2.0128, Cate Loss: 0.3902, Mask Loss: 0.5409
Epoch [5/36], Iteration [20/40], Loss: 2.7772, Cate Loss: 0.3901, Mask Loss: 0.7957
Epoch [5/36], Iteration [30/40], Loss: 1.4085, Cate Loss: 0.3902, Mask Loss: 0.3394
Epoch [5/36] finished, Average Loss: 1.9396
Checkpoint saved at ./checkpoints/checkpoint_epoch_5.pth
Epoch [6/36], Iteration [0/40], Loss: 1.3728, Cate Loss: 0.3903, Mask Loss: 0.3275
Epoch [6/36], Iteration [10/40], Loss: 2.8399, Cate Loss: 0.3901, Mask Loss: 0.8166
Epoch [6/36], Iteration [20/40], Loss: 1.4178, Cate Loss: 0.3901, Mask Loss: 0.3426
Epoch [6/36], Iteration [30/40], Loss: 2.0569, Cate Loss: 0.3899, Mask Loss: 0.5557
Epoch [6/36] finished, Average Loss: 1.8585
Checkpoint saved at ./checkpoints/checkpoint_epoch_6.pth
Epoch [7/36], Iteration [0/40], Loss: 2.8822, Cate Loss: 0.3904, Mask Loss: 0.8306
Epoch [7/36], Iteration [10/40], Loss: 1.5060, Cate Loss: 0.3898, Mask Loss: 0.3721
Epoch [7/36], Iteration [20/40], Loss: 1.2449, Cate Loss: 0.3900, Mask Loss: 0.2850
Epoch [7/36], Iteration [30/40], Loss: 1.6769, Cate Loss: 0.3902, Mask Loss: 0.4289
Epoch [7/36] finished, Average Loss: 1.8095
Checkpoint saved at ./checkpoints/checkpoint_epoch_7.pth
Epoch [8/36], Iteration [0/40], Loss: 2.4946, Cate Loss: 0.3902, Mask Loss: 0.7015
Epoch [8/36], Iteration [10/40], Loss: 1.4369, Cate Loss: 0.3902, Mask Loss: 0.3489
Epoch [8/36], Iteration [20/40], Loss: 1.9396, Cate Loss: 0.3896, Mask Loss: 0.5167
Epoch [8/36], Iteration [30/40], Loss: 1.1262, Cate Loss: 0.3900, Mask Loss: 0.2454
Epoch [8/36] finished, Average Loss: 1.7239
Checkpoint saved at ./checkpoints/checkpoint_epoch_8.pth
Epoch [9/36], Iteration [0/40], Loss: 0.6926, Cate Loss: 0.3901, Mask Loss: 0.1008
Epoch [9/36], Iteration [10/40], Loss: 1.8026, Cate Loss: 0.3897, Mask Loss: 0.4710
Epoch [9/36], Iteration [20/40], Loss: 2.0997, Cate Loss: 0.3895, Mask Loss: 0.5701
Epoch [9/36], Iteration [30/40], Loss: 2.9913, Cate Loss: 0.3901, Mask Loss: 0.8671
Epoch [9/36] finished, Average Loss: 1.6966
Checkpoint saved at ./checkpoints/checkpoint_epoch_9.pth
Epoch [10/36], Iteration [0/40], Loss: 2.3688, Cate Loss: 0.3899, Mask Loss: 0.6596
Epoch [10/36], Iteration [10/40], Loss: 1.9495, Cate Loss: 0.3897, Mask Loss: 0.5199
Epoch [10/36], Iteration [20/40], Loss: 2.1152, Cate Loss: 0.3901, Mask Loss: 0.5751
Epoch [10/36], Iteration [30/40], Loss: 1.5407, Cate Loss: 0.3896, Mask Loss: 0.3837
Epoch [10/36] finished, Average Loss: 1.5771
Checkpoint saved at ./checkpoints/checkpoint_epoch_10.pth
Epoch [11/36], Iteration [0/40], Loss: 1.1559, Cate Loss: 0.3895, Mask Loss: 0.2555
Epoch [11/36], Iteration [10/40], Loss: 1.6556, Cate Loss: 0.3899, Mask Loss: 0.4219
Epoch [11/36], Iteration [20/40], Loss: 1.3084, Cate Loss: 0.3898, Mask Loss: 0.3062
Epoch [11/36], Iteration [30/40], Loss: 1.5152, Cate Loss: 0.3900, Mask Loss: 0.3751
Epoch [11/36] finished, Average Loss: 1.5565
Checkpoint saved at ./checkpoints/checkpoint_epoch_11.pth
Epoch [12/36], Iteration [0/40], Loss: 0.6363, Cate Loss: 0.3901, Mask Loss: 0.0821
Epoch [12/36], Iteration [10/40], Loss: 1.8941, Cate Loss: 0.3900, Mask Loss: 0.5014
Epoch [12/36], Iteration [20/40], Loss: 1.2148, Cate Loss: 0.3899, Mask Loss: 0.2750
Epoch [12/36], Iteration [30/40], Loss: 1.8123, Cate Loss: 0.3898, Mask Loss: 0.4742
Epoch [12/36] finished, Average Loss: 1.4863
Checkpoint saved at ./checkpoints/checkpoint_epoch_12.pth
Epoch [13/36], Iteration [0/40], Loss: 0.7670, Cate Loss: 0.3899, Mask Loss: 0.1257
Epoch [13/36], Iteration [10/40], Loss: 0.7210, Cate Loss: 0.3898, Mask Loss: 0.1104
Epoch [13/36], Iteration [20/40], Loss: 1.6621, Cate Loss: 0.3896, Mask Loss: 0.4242
Epoch [13/36], Iteration [30/40], Loss: 1.9835, Cate Loss: 0.3896, Mask Loss: 0.5313
Epoch [13/36] finished, Average Loss: 1.4300
Checkpoint saved at ./checkpoints/checkpoint_epoch_13.pth
Epoch [14/36], Iteration [0/40], Loss: 1.1037, Cate Loss: 0.3894, Mask Loss: 0.2381
Epoch [14/36], Iteration [10/40], Loss: 1.7021, Cate Loss: 0.3894, Mask Loss: 0.4376
Epoch [14/36], Iteration [20/40], Loss: 1.4994, Cate Loss: 0.3899, Mask Loss: 0.3698
Epoch [14/36], Iteration [30/40], Loss: 1.2821, Cate Loss: 0.3893, Mask Loss: 0.2976
Epoch [14/36] finished, Average Loss: 1.4143
Checkpoint saved at ./checkpoints/checkpoint_epoch_14.pth
Epoch [15/36], Iteration [0/40], Loss: 2.0415, Cate Loss: 0.3897, Mask Loss: 0.5506
Epoch [15/36], Iteration [10/40], Loss: 2.3190, Cate Loss: 0.3899, Mask Loss: 0.6430
Epoch [15/36], Iteration [20/40], Loss: 2.0824, Cate Loss: 0.3892, Mask Loss: 0.5644
Epoch [15/36], Iteration [30/40], Loss: 1.1222, Cate Loss: 0.3897, Mask Loss: 0.2442
Epoch [15/36] finished, Average Loss: 1.3630
Checkpoint saved at ./checkpoints/checkpoint_epoch_15.pth
Epoch [16/36], Iteration [0/40], Loss: 1.2776, Cate Loss: 0.3894, Mask Loss: 0.2961
Epoch [16/36], Iteration [10/40], Loss: 1.7055, Cate Loss: 0.3897, Mask Loss: 0.4386
Epoch [16/36], Iteration [20/40], Loss: 1.7159, Cate Loss: 0.3898, Mask Loss: 0.4420
Epoch [16/36], Iteration [30/40], Loss: 1.1734, Cate Loss: 0.3894, Mask Loss: 0.2614
Epoch [16/36] finished, Average Loss: 1.3334
Checkpoint saved at ./checkpoints/checkpoint_epoch_16.pth
Epoch [17/36], Iteration [0/40], Loss: 0.5328, Cate Loss: 0.3895, Mask Loss: 0.0478
Epoch [17/36], Iteration [10/40], Loss: 1.5855, Cate Loss: 0.3899, Mask Loss: 0.3986
Epoch [17/36], Iteration [20/40], Loss: 2.0702, Cate Loss: 0.3899, Mask Loss: 0.5601
Epoch [17/36], Iteration [30/40], Loss: 0.6934, Cate Loss: 0.3898, Mask Loss: 0.1012
Epoch [17/36] finished, Average Loss: 1.3005
Checkpoint saved at ./checkpoints/checkpoint_epoch_17.pth
Epoch [18/36], Iteration [0/40], Loss: 1.7363, Cate Loss: 0.3894, Mask Loss: 0.4490
Epoch [18/36], Iteration [10/40], Loss: 0.7421, Cate Loss: 0.3897, Mask Loss: 0.1175
Epoch [18/36], Iteration [20/40], Loss: 1.9736, Cate Loss: 0.3899, Mask Loss: 0.5279
Epoch [18/36], Iteration [30/40], Loss: 0.8929, Cate Loss: 0.3896, Mask Loss: 0.1678
Epoch [18/36] finished, Average Loss: 1.2999
Checkpoint saved at ./checkpoints/checkpoint_epoch_18.pth
Epoch [19/36], Iteration [0/40], Loss: 1.2661, Cate Loss: 0.3896, Mask Loss: 0.2922
Epoch [19/36], Iteration [10/40], Loss: 1.7355, Cate Loss: 0.3895, Mask Loss: 0.4487
Epoch [19/36], Iteration [20/40], Loss: 1.0973, Cate Loss: 0.3897, Mask Loss: 0.2359
Epoch [19/36], Iteration [30/40], Loss: 1.1334, Cate Loss: 0.3897, Mask Loss: 0.2479
Epoch [19/36] finished, Average Loss: 1.2509
Checkpoint saved at ./checkpoints/checkpoint_epoch_19.pth
Epoch [20/36], Iteration [0/40], Loss: 1.8637, Cate Loss: 0.3896, Mask Loss: 0.4914
Epoch [20/36], Iteration [10/40], Loss: 0.6480, Cate Loss: 0.3897, Mask Loss: 0.0861
Epoch [20/36], Iteration [20/40], Loss: 2.1740, Cate Loss: 0.3894, Mask Loss: 0.5948
Epoch [20/36], Iteration [30/40], Loss: 2.0820, Cate Loss: 0.3897, Mask Loss: 0.5641
Epoch [20/36] finished, Average Loss: 1.2375
Checkpoint saved at ./checkpoints/checkpoint_epoch_20.pth
Epoch [21/36], Iteration [0/40], Loss: 1.2566, Cate Loss: 0.3894, Mask Loss: 0.2891
Epoch [21/36], Iteration [10/40], Loss: 1.3972, Cate Loss: 0.3895, Mask Loss: 0.3359
Epoch [21/36], Iteration [20/40], Loss: 1.7520, Cate Loss: 0.3898, Mask Loss: 0.4540
Epoch [21/36], Iteration [30/40], Loss: 1.5731, Cate Loss: 0.3896, Mask Loss: 0.3945
Epoch [21/36] finished, Average Loss: 1.2325
Checkpoint saved at ./checkpoints/checkpoint_epoch_21.pth
Epoch [22/36], Iteration [0/40], Loss: 0.6295, Cate Loss: 0.3896, Mask Loss: 0.0799
Epoch [22/36], Iteration [10/40], Loss: 1.8894, Cate Loss: 0.3897, Mask Loss: 0.4999
Epoch [22/36], Iteration [20/40], Loss: 0.7807, Cate Loss: 0.3896, Mask Loss: 0.1304
Epoch [22/36], Iteration [30/40], Loss: 1.1767, Cate Loss: 0.3893, Mask Loss: 0.2625
Epoch [22/36] finished, Average Loss: 1.2080
Checkpoint saved at ./checkpoints/checkpoint_epoch_22.pth
Epoch [23/36], Iteration [0/40], Loss: 1.0928, Cate Loss: 0.3897, Mask Loss: 0.2344
Epoch [23/36], Iteration [10/40], Loss: 0.7776, Cate Loss: 0.3898, Mask Loss: 0.1293
Epoch [23/36], Iteration [20/40], Loss: 1.4354, Cate Loss: 0.3897, Mask Loss: 0.3486
Epoch [23/36], Iteration [30/40], Loss: 1.2260, Cate Loss: 0.3895, Mask Loss: 0.2788
Epoch [23/36] finished, Average Loss: 1.2016
Checkpoint saved at ./checkpoints/checkpoint_epoch_23.pth
Epoch [24/36], Iteration [0/40], Loss: 2.5826, Cate Loss: 0.3898, Mask Loss: 0.7309
Epoch [24/36], Iteration [10/40], Loss: 1.0140, Cate Loss: 0.3897, Mask Loss: 0.2081
Epoch [24/36], Iteration [20/40], Loss: 0.7414, Cate Loss: 0.3896, Mask Loss: 0.1172
Epoch [24/36], Iteration [30/40], Loss: 1.4034, Cate Loss: 0.3891, Mask Loss: 0.3381
Epoch [24/36] finished, Average Loss: 1.1699
Checkpoint saved at ./checkpoints/checkpoint_epoch_24.pth
Epoch [25/36], Iteration [0/40], Loss: 1.4735, Cate Loss: 0.3897, Mask Loss: 0.3613
Epoch [25/36], Iteration [10/40], Loss: 2.3163, Cate Loss: 0.3897, Mask Loss: 0.6422
Epoch [25/36], Iteration [20/40], Loss: 0.6783, Cate Loss: 0.3896, Mask Loss: 0.0962
Epoch [25/36], Iteration [30/40], Loss: 0.8254, Cate Loss: 0.3895, Mask Loss: 0.1453
Epoch [25/36] finished, Average Loss: 1.1461
Checkpoint saved at ./checkpoints/checkpoint_epoch_25.pth
Epoch [26/36], Iteration [0/40], Loss: 2.0371, Cate Loss: 0.3896, Mask Loss: 0.5492
Epoch [26/36], Iteration [10/40], Loss: 0.7458, Cate Loss: 0.3898, Mask Loss: 0.1187
Epoch [26/36], Iteration [20/40], Loss: 1.2198, Cate Loss: 0.3894, Mask Loss: 0.2768
Epoch [26/36], Iteration [30/40], Loss: 1.2772, Cate Loss: 0.3891, Mask Loss: 0.2960
Epoch [26/36] finished, Average Loss: 1.1372
Checkpoint saved at ./checkpoints/checkpoint_epoch_26.pth
Epoch [27/36], Iteration [0/40], Loss: 1.2502, Cate Loss: 0.3895, Mask Loss: 0.2869
Epoch [27/36], Iteration [10/40], Loss: 0.6343, Cate Loss: 0.3896, Mask Loss: 0.0816
Epoch [27/36], Iteration [20/40], Loss: 1.0810, Cate Loss: 0.3892, Mask Loss: 0.2306
Epoch [27/36], Iteration [30/40], Loss: 1.9459, Cate Loss: 0.3898, Mask Loss: 0.5187
Epoch [27/36] finished, Average Loss: 1.1349
Checkpoint saved at ./checkpoints/checkpoint_epoch_27.pth
Epoch [28/36], Iteration [0/40], Loss: 1.3822, Cate Loss: 0.3893, Mask Loss: 0.3309
Epoch [28/36], Iteration [10/40], Loss: 0.7555, Cate Loss: 0.3898, Mask Loss: 0.1219
Epoch [28/36], Iteration [20/40], Loss: 1.1647, Cate Loss: 0.3895, Mask Loss: 0.2584
Epoch [28/36], Iteration [30/40], Loss: 1.0837, Cate Loss: 0.3895, Mask Loss: 0.2314
Epoch [28/36] finished, Average Loss: 1.0877
Checkpoint saved at ./checkpoints/checkpoint_epoch_28.pth
Epoch [29/36], Iteration [0/40], Loss: 1.5358, Cate Loss: 0.3895, Mask Loss: 0.3821
Epoch [29/36], Iteration [10/40], Loss: 2.1581, Cate Loss: 0.3897, Mask Loss: 0.5894
Epoch [29/36], Iteration [20/40], Loss: 0.7221, Cate Loss: 0.3894, Mask Loss: 0.1109
Epoch [29/36], Iteration [30/40], Loss: 1.2733, Cate Loss: 0.3893, Mask Loss: 0.2946
Epoch [29/36] finished, Average Loss: 1.0913
Checkpoint saved at ./checkpoints/checkpoint_epoch_29.pth
Epoch [30/36], Iteration [0/40], Loss: 0.5989, Cate Loss: 0.3896, Mask Loss: 0.0698
Epoch [30/36], Iteration [10/40], Loss: 0.8878, Cate Loss: 0.3897, Mask Loss: 0.1660
Epoch [30/36], Iteration [20/40], Loss: 0.6313, Cate Loss: 0.3895, Mask Loss: 0.0806
Epoch [30/36], Iteration [30/40], Loss: 0.8770, Cate Loss: 0.3897, Mask Loss: 0.1624
Epoch [30/36] finished, Average Loss: 1.0732
Checkpoint saved at ./checkpoints/checkpoint_epoch_30.pth
Epoch [31/36], Iteration [0/40], Loss: 0.8200, Cate Loss: 0.3897, Mask Loss: 0.1434
Epoch [31/36], Iteration [10/40], Loss: 1.8109, Cate Loss: 0.3897, Mask Loss: 0.4737
Epoch [31/36], Iteration [20/40], Loss: 1.0111, Cate Loss: 0.3896, Mask Loss: 0.2072
Epoch [31/36], Iteration [30/40], Loss: 0.9308, Cate Loss: 0.3896, Mask Loss: 0.1804
Epoch [31/36] finished, Average Loss: 1.0587
Checkpoint saved at ./checkpoints/checkpoint_epoch_31.pth
Epoch [32/36], Iteration [0/40], Loss: 0.9318, Cate Loss: 0.3893, Mask Loss: 0.1808
Epoch [32/36], Iteration [10/40], Loss: 0.5219, Cate Loss: 0.3893, Mask Loss: 0.0442
Epoch [32/36], Iteration [20/40], Loss: 0.7513, Cate Loss: 0.3894, Mask Loss: 0.1206
Epoch [32/36], Iteration [30/40], Loss: 1.0362, Cate Loss: 0.3892, Mask Loss: 0.2156
Epoch [32/36] finished, Average Loss: 1.0911
Checkpoint saved at ./checkpoints/checkpoint_epoch_32.pth
Epoch [33/36], Iteration [0/40], Loss: 0.6599, Cate Loss: 0.3896, Mask Loss: 0.0901
Epoch [33/36], Iteration [10/40], Loss: 0.9272, Cate Loss: 0.3896, Mask Loss: 0.1792
Epoch [33/36], Iteration [20/40], Loss: 0.9590, Cate Loss: 0.3896, Mask Loss: 0.1898
Epoch [33/36], Iteration [30/40], Loss: 1.2359, Cate Loss: 0.3894, Mask Loss: 0.2822
Epoch [33/36] finished, Average Loss: 1.0647
Checkpoint saved at ./checkpoints/checkpoint_epoch_33.pth
Epoch [34/36], Iteration [0/40], Loss: 2.3724, Cate Loss: 0.3897, Mask Loss: 0.6609
Epoch [34/36], Iteration [10/40], Loss: 1.4618, Cate Loss: 0.3898, Mask Loss: 0.3573
Epoch [34/36], Iteration [20/40], Loss: 0.5930, Cate Loss: 0.3894, Mask Loss: 0.0679
Epoch [34/36], Iteration [30/40], Loss: 0.8167, Cate Loss: 0.3894, Mask Loss: 0.1424
Epoch [34/36] finished, Average Loss: 1.0831
Checkpoint saved at ./checkpoints/checkpoint_epoch_34.pth
Epoch [35/36], Iteration [0/40], Loss: 0.9642, Cate Loss: 0.3893, Mask Loss: 0.1916
Epoch [35/36], Iteration [10/40], Loss: 0.5041, Cate Loss: 0.3896, Mask Loss: 0.0381
Epoch [35/36], Iteration [20/40], Loss: 1.1844, Cate Loss: 0.3892, Mask Loss: 0.2651
Epoch [35/36], Iteration [30/40], Loss: 0.5838, Cate Loss: 0.3895, Mask Loss: 0.0648
Epoch [35/36] finished, Average Loss: 1.0723
Checkpoint saved at ./checkpoints/checkpoint_epoch_35.pth
Epoch [36/36], Iteration [0/40], Loss: 1.6185, Cate Loss: 0.3896, Mask Loss: 0.4096
Epoch [36/36], Iteration [10/40], Loss: 1.3159, Cate Loss: 0.3896, Mask Loss: 0.3088
Epoch [36/36], Iteration [20/40], Loss: 1.5603, Cate Loss: 0.3896, Mask Loss: 0.3902
Epoch [36/36], Iteration [30/40], Loss: 0.6219, Cate Loss: 0.3894, Mask Loss: 0.0775
Epoch [36/36] finished, Average Loss: 1.0764
Checkpoint saved at ./checkpoints/checkpoint_epoch_36.pth
Training complete.