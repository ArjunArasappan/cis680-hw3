import re
import matplotlib.pyplot as plt

# Paste your logs as a multi-line string
logs = """
Using device: cuda
Epoch [1/100], Iteration [0/35], Loss: 2.8565, Cate Loss: 0.5119, Mask Loss: 0.7815
Epoch [1/100], Iteration [10/35], Loss: 1.7064, Cate Loss: 0.0116, Mask Loss: 0.5649
Epoch [1/100], Iteration [20/35], Loss: 1.8689, Cate Loss: 0.0184, Mask Loss: 0.6169
Epoch [1/100], Iteration [30/35], Loss: 1.7918, Cate Loss: 0.0130, Mask Loss: 0.5929
Epoch [1/100] finished, Average Training Loss: 2.1275
Epoch [1/100] finished, Average Validation Loss: 2.3358
Checkpoint saved at ./checkpoints/checkpoint_epoch_1.pth
Epoch [2/100], Iteration [0/35], Loss: 2.1654, Cate Loss: 0.0107, Mask Loss: 0.7182
Epoch [2/100], Iteration [10/35], Loss: 1.8641, Cate Loss: 0.0125, Mask Loss: 0.6172
Epoch [2/100], Iteration [20/35], Loss: 1.8733, Cate Loss: 0.0116, Mask Loss: 0.6206
Epoch [2/100], Iteration [30/35], Loss: 2.0242, Cate Loss: 0.0105, Mask Loss: 0.6713
Epoch [2/100] finished, Average Training Loss: 1.9454
Epoch [2/100] finished, Average Validation Loss: 2.2171
Checkpoint saved at ./checkpoints/checkpoint_epoch_2.pth
Epoch [3/100], Iteration [0/35], Loss: 1.1045, Cate Loss: 0.0142, Mask Loss: 0.3634
Epoch [3/100], Iteration [10/35], Loss: 0.8223, Cate Loss: 0.0131, Mask Loss: 0.2697
Epoch [3/100], Iteration [20/35], Loss: 1.6828, Cate Loss: 0.0151, Mask Loss: 0.5559
Epoch [3/100], Iteration [30/35], Loss: 1.7849, Cate Loss: 0.0104, Mask Loss: 0.5915
Epoch [3/100] finished, Average Training Loss: 1.7765
Epoch [3/100] finished, Average Validation Loss: 2.0995
Checkpoint saved at ./checkpoints/checkpoint_epoch_3.pth
Epoch [4/100], Iteration [0/35], Loss: 1.7552, Cate Loss: 0.0182, Mask Loss: 0.5790
Epoch [4/100], Iteration [10/35], Loss: 1.7910, Cate Loss: 0.0144, Mask Loss: 0.5922
Epoch [4/100], Iteration [20/35], Loss: 0.9913, Cate Loss: 0.0135, Mask Loss: 0.3260
Epoch [4/100], Iteration [30/35], Loss: 1.8298, Cate Loss: 0.0091, Mask Loss: 0.6069
Epoch [4/100] finished, Average Training Loss: 1.6618
Epoch [4/100] finished, Average Validation Loss: 2.0176
Checkpoint saved at ./checkpoints/checkpoint_epoch_4.pth
Epoch [5/100], Iteration [0/35], Loss: 1.4482, Cate Loss: 0.0085, Mask Loss: 0.4799
Epoch [5/100], Iteration [10/35], Loss: 1.9161, Cate Loss: 0.0087, Mask Loss: 0.6358
Epoch [5/100], Iteration [20/35], Loss: 2.3617, Cate Loss: 0.0131, Mask Loss: 0.7828
Epoch [5/100], Iteration [30/35], Loss: 1.5390, Cate Loss: 0.0082, Mask Loss: 0.5103
Epoch [5/100] finished, Average Training Loss: 1.5549
Epoch [5/100] finished, Average Validation Loss: 1.9256
Checkpoint saved at ./checkpoints/checkpoint_epoch_5.pth
Epoch [6/100], Iteration [0/35], Loss: 1.7227, Cate Loss: 0.0053, Mask Loss: 0.5725
Epoch [6/100], Iteration [10/35], Loss: 0.7194, Cate Loss: 0.0084, Mask Loss: 0.2370
Epoch [6/100], Iteration [20/35], Loss: 2.4313, Cate Loss: 0.0059, Mask Loss: 0.8085
Epoch [6/100], Iteration [30/35], Loss: 1.3099, Cate Loss: 0.0079, Mask Loss: 0.4340
Epoch [6/100] finished, Average Training Loss: 1.5152
Epoch [6/100] finished, Average Validation Loss: 1.9067
Checkpoint saved at ./checkpoints/checkpoint_epoch_6.pth
Epoch [7/100], Iteration [0/35], Loss: 1.3663, Cate Loss: 0.0062, Mask Loss: 0.4534
Epoch [7/100], Iteration [10/35], Loss: 2.0634, Cate Loss: 0.0058, Mask Loss: 0.6859
Epoch [7/100], Iteration [20/35], Loss: 1.5517, Cate Loss: 0.0107, Mask Loss: 0.5137
Epoch [7/100], Iteration [30/35], Loss: 1.9380, Cate Loss: 0.0043, Mask Loss: 0.6446
Epoch [7/100] finished, Average Training Loss: 1.4591
Epoch [7/100] finished, Average Validation Loss: 1.8549
Checkpoint saved at ./checkpoints/checkpoint_epoch_7.pth
Epoch [8/100], Iteration [0/35], Loss: 0.9264, Cate Loss: 0.0077, Mask Loss: 0.3062
Epoch [8/100], Iteration [10/35], Loss: 1.5667, Cate Loss: 0.0026, Mask Loss: 0.5214
Epoch [8/100], Iteration [20/35], Loss: 1.0151, Cate Loss: 0.0075, Mask Loss: 0.3359
Epoch [8/100], Iteration [30/35], Loss: 1.5750, Cate Loss: 0.0027, Mask Loss: 0.5241
Epoch [8/100] finished, Average Training Loss: 1.4717
Epoch [8/100] finished, Average Validation Loss: 1.8070
Checkpoint saved at ./checkpoints/checkpoint_epoch_8.pth
Epoch [9/100], Iteration [0/35], Loss: 0.9512, Cate Loss: 0.0036, Mask Loss: 0.3159
Epoch [9/100], Iteration [10/35], Loss: 1.1638, Cate Loss: 0.0037, Mask Loss: 0.3867
Epoch [9/100], Iteration [20/35], Loss: 1.0778, Cate Loss: 0.0045, Mask Loss: 0.3578
Epoch [9/100], Iteration [30/35], Loss: 0.9608, Cate Loss: 0.0067, Mask Loss: 0.3181
Epoch [9/100] finished, Average Training Loss: 1.3330
Epoch [9/100] finished, Average Validation Loss: 1.7367
Checkpoint saved at ./checkpoints/checkpoint_epoch_9.pth
Epoch [10/100], Iteration [0/35], Loss: 0.7338, Cate Loss: 0.0030, Mask Loss: 0.2436
Epoch [10/100], Iteration [10/35], Loss: 2.1954, Cate Loss: 0.0031, Mask Loss: 0.7308
Epoch [10/100], Iteration [20/35], Loss: 0.9752, Cate Loss: 0.0041, Mask Loss: 0.3237
Epoch [10/100], Iteration [30/35], Loss: 0.8276, Cate Loss: 0.0059, Mask Loss: 0.2739
Epoch [10/100] finished, Average Training Loss: 1.3297
Epoch [10/100] finished, Average Validation Loss: 1.6899
Checkpoint saved at ./checkpoints/checkpoint_epoch_10.pth
Epoch [11/100], Iteration [0/35], Loss: 1.6438, Cate Loss: 0.0063, Mask Loss: 0.5458
Epoch [11/100], Iteration [10/35], Loss: 1.1793, Cate Loss: 0.0061, Mask Loss: 0.3911
Epoch [11/100], Iteration [20/35], Loss: 0.5033, Cate Loss: 0.0024, Mask Loss: 0.1670
Epoch [11/100], Iteration [30/35], Loss: 0.9636, Cate Loss: 0.0058, Mask Loss: 0.3192
Epoch [11/100] finished, Average Training Loss: 1.2480
Epoch [11/100] finished, Average Validation Loss: 1.6650
Checkpoint saved at ./checkpoints/checkpoint_epoch_11.pth
Epoch [12/100], Iteration [0/35], Loss: 2.0873, Cate Loss: 0.0025, Mask Loss: 0.6949
Epoch [12/100], Iteration [10/35], Loss: 0.6162, Cate Loss: 0.0029, Mask Loss: 0.2044
Epoch [12/100], Iteration [20/35], Loss: 0.8064, Cate Loss: 0.0036, Mask Loss: 0.2676
Epoch [12/100], Iteration [30/35], Loss: 0.9287, Cate Loss: 0.0022, Mask Loss: 0.3088
Epoch [12/100] finished, Average Training Loss: 1.1773
Epoch [12/100] finished, Average Validation Loss: 1.6219
Checkpoint saved at ./checkpoints/checkpoint_epoch_12.pth
Epoch [13/100], Iteration [0/35], Loss: 1.8362, Cate Loss: 0.0030, Mask Loss: 0.6111
Epoch [13/100], Iteration [10/35], Loss: 1.2304, Cate Loss: 0.0033, Mask Loss: 0.4090
Epoch [13/100], Iteration [20/35], Loss: 2.0005, Cate Loss: 0.0028, Mask Loss: 0.6659
Epoch [13/100], Iteration [30/35], Loss: 1.7010, Cate Loss: 0.0047, Mask Loss: 0.5654
Epoch [13/100] finished, Average Training Loss: 1.1071
Epoch [13/100] finished, Average Validation Loss: 1.6159
Checkpoint saved at ./checkpoints/checkpoint_epoch_13.pth
Epoch [14/100], Iteration [0/35], Loss: 1.0952, Cate Loss: 0.0041, Mask Loss: 0.3637
Epoch [14/100], Iteration [10/35], Loss: 0.6612, Cate Loss: 0.0030, Mask Loss: 0.2194
Epoch [14/100], Iteration [20/35], Loss: 1.5591, Cate Loss: 0.0030, Mask Loss: 0.5187
Epoch [14/100], Iteration [30/35], Loss: 1.5656, Cate Loss: 0.0033, Mask Loss: 0.5208
Epoch [14/100] finished, Average Training Loss: 1.0768
Epoch [14/100] finished, Average Validation Loss: 1.5735
Checkpoint saved at ./checkpoints/checkpoint_epoch_14.pth
Epoch [15/100], Iteration [0/35], Loss: 1.2371, Cate Loss: 0.0024, Mask Loss: 0.4116
Epoch [15/100], Iteration [10/35], Loss: 1.8171, Cate Loss: 0.0025, Mask Loss: 0.6049
Epoch [15/100], Iteration [20/35], Loss: 0.7616, Cate Loss: 0.0031, Mask Loss: 0.2528
Epoch [15/100], Iteration [30/35], Loss: 0.9157, Cate Loss: 0.0017, Mask Loss: 0.3047
Epoch [15/100] finished, Average Training Loss: 1.0160
Epoch [15/100] finished, Average Validation Loss: 1.6020
Checkpoint saved at ./checkpoints/checkpoint_epoch_15.pth
Epoch [16/100], Iteration [0/35], Loss: 0.4416, Cate Loss: 0.0023, Mask Loss: 0.1464
Epoch [16/100], Iteration [10/35], Loss: 1.1389, Cate Loss: 0.0030, Mask Loss: 0.3786
Epoch [16/100], Iteration [20/35], Loss: 1.4145, Cate Loss: 0.0027, Mask Loss: 0.4706
Epoch [16/100], Iteration [30/35], Loss: 1.2460, Cate Loss: 0.0024, Mask Loss: 0.4145
Epoch [16/100] finished, Average Training Loss: 0.9778
Epoch [16/100] finished, Average Validation Loss: 1.5489
Checkpoint saved at ./checkpoints/checkpoint_epoch_16.pth
Epoch [17/100], Iteration [0/35], Loss: 0.8908, Cate Loss: 0.0029, Mask Loss: 0.2960
Epoch [17/100], Iteration [10/35], Loss: 1.2256, Cate Loss: 0.0028, Mask Loss: 0.4076
Epoch [17/100], Iteration [20/35], Loss: 0.9812, Cate Loss: 0.0027, Mask Loss: 0.3262
Epoch [17/100], Iteration [30/35], Loss: 1.0767, Cate Loss: 0.0023, Mask Loss: 0.3581
Epoch [17/100] finished, Average Training Loss: 0.9558
Epoch [17/100] finished, Average Validation Loss: 1.5516
Checkpoint saved at ./checkpoints/checkpoint_epoch_17.pth
Epoch [18/100], Iteration [0/35], Loss: 2.1017, Cate Loss: 0.0031, Mask Loss: 0.6995
Epoch [18/100], Iteration [10/35], Loss: 1.3139, Cate Loss: 0.0024, Mask Loss: 0.4372
Epoch [18/100], Iteration [20/35], Loss: 0.4224, Cate Loss: 0.0021, Mask Loss: 0.1401
Epoch [18/100], Iteration [30/35], Loss: 0.5582, Cate Loss: 0.0018, Mask Loss: 0.1854
Epoch [18/100] finished, Average Training Loss: 0.9424
Epoch [18/100] finished, Average Validation Loss: 1.5033
Checkpoint saved at ./checkpoints/checkpoint_epoch_18.pth
Epoch [19/100], Iteration [0/35], Loss: 0.8217, Cate Loss: 0.0031, Mask Loss: 0.2729
Epoch [19/100], Iteration [10/35], Loss: 0.7446, Cate Loss: 0.0043, Mask Loss: 0.2468
Epoch [19/100], Iteration [20/35], Loss: 0.7927, Cate Loss: 0.0011, Mask Loss: 0.2639
Epoch [19/100], Iteration [30/35], Loss: 0.2770, Cate Loss: 0.0021, Mask Loss: 0.0916
Epoch [19/100] finished, Average Training Loss: 0.9334
Epoch [19/100] finished, Average Validation Loss: 1.4961
Checkpoint saved at ./checkpoints/checkpoint_epoch_19.pth
Epoch [20/100], Iteration [0/35], Loss: 0.7130, Cate Loss: 0.0029, Mask Loss: 0.2367
Epoch [20/100], Iteration [10/35], Loss: 0.7134, Cate Loss: 0.0013, Mask Loss: 0.2374
Epoch [20/100], Iteration [20/35], Loss: 1.4596, Cate Loss: 0.0020, Mask Loss: 0.4858
Epoch [20/100], Iteration [30/35], Loss: 0.9023, Cate Loss: 0.0014, Mask Loss: 0.3003
Epoch [20/100] finished, Average Training Loss: 0.8783
Epoch [20/100] finished, Average Validation Loss: 1.4385
Checkpoint saved at ./checkpoints/checkpoint_epoch_20.pth
Epoch [21/100], Iteration [0/35], Loss: 0.6623, Cate Loss: 0.0042, Mask Loss: 0.2194
Epoch [21/100], Iteration [10/35], Loss: 0.3766, Cate Loss: 0.0015, Mask Loss: 0.1250
Epoch [21/100], Iteration [20/35], Loss: 1.4621, Cate Loss: 0.0027, Mask Loss: 0.4865
Epoch [21/100], Iteration [30/35], Loss: 0.8754, Cate Loss: 0.0018, Mask Loss: 0.2912
Epoch [21/100] finished, Average Training Loss: 0.8499
Epoch [21/100] finished, Average Validation Loss: 1.4584
Checkpoint saved at ./checkpoints/checkpoint_epoch_21.pth
Epoch [22/100], Iteration [0/35], Loss: 0.9354, Cate Loss: 0.0030, Mask Loss: 0.3108
Epoch [22/100], Iteration [10/35], Loss: 0.2516, Cate Loss: 0.0020, Mask Loss: 0.0832
Epoch [22/100], Iteration [20/35], Loss: 0.7778, Cate Loss: 0.0022, Mask Loss: 0.2586
Epoch [22/100], Iteration [30/35], Loss: 1.4562, Cate Loss: 0.0027, Mask Loss: 0.4845
Epoch [22/100] finished, Average Training Loss: 0.8396
Epoch [22/100] finished, Average Validation Loss: 1.4451
Checkpoint saved at ./checkpoints/checkpoint_epoch_22.pth
Epoch [23/100], Iteration [0/35], Loss: 1.6369, Cate Loss: 0.0029, Mask Loss: 0.5447
Epoch [23/100], Iteration [10/35], Loss: 0.4949, Cate Loss: 0.0024, Mask Loss: 0.1642
Epoch [23/100], Iteration [20/35], Loss: 0.5421, Cate Loss: 0.0027, Mask Loss: 0.1798
Epoch [23/100], Iteration [30/35], Loss: 1.2047, Cate Loss: 0.0025, Mask Loss: 0.4007
Epoch [23/100] finished, Average Training Loss: 0.8444
Epoch [23/100] finished, Average Validation Loss: 1.5114
Checkpoint saved at ./checkpoints/checkpoint_epoch_23.pth
Epoch [24/100], Iteration [0/35], Loss: 0.2801, Cate Loss: 0.0024, Mask Loss: 0.0926
Epoch [24/100], Iteration [10/35], Loss: 1.5029, Cate Loss: 0.0028, Mask Loss: 0.5000
Epoch [24/100], Iteration [20/35], Loss: 0.3450, Cate Loss: 0.0024, Mask Loss: 0.1142
Epoch [24/100], Iteration [30/35], Loss: 0.2680, Cate Loss: 0.0023, Mask Loss: 0.0885
Epoch [24/100] finished, Average Training Loss: 0.8271
Epoch [24/100] finished, Average Validation Loss: 1.5237
Checkpoint saved at ./checkpoints/checkpoint_epoch_24.pth
Epoch [25/100], Iteration [0/35], Loss: 0.2907, Cate Loss: 0.0026, Mask Loss: 0.0961
Epoch [25/100], Iteration [10/35], Loss: 1.3503, Cate Loss: 0.0024, Mask Loss: 0.4493
Epoch [25/100], Iteration [20/35], Loss: 0.2514, Cate Loss: 0.0022, Mask Loss: 0.0831
Epoch [25/100], Iteration [30/35], Loss: 0.1716, Cate Loss: 0.0019, Mask Loss: 0.0566
Epoch [25/100] finished, Average Training Loss: 0.7829
Epoch [25/100] finished, Average Validation Loss: 1.4356
Checkpoint saved at ./checkpoints/checkpoint_epoch_25.pth
Epoch [26/100], Iteration [0/35], Loss: 0.3811, Cate Loss: 0.0014, Mask Loss: 0.1266
Epoch [26/100], Iteration [10/35], Loss: 0.8550, Cate Loss: 0.0018, Mask Loss: 0.2844
Epoch [26/100], Iteration [20/35], Loss: 1.3751, Cate Loss: 0.0026, Mask Loss: 0.4575
Epoch [26/100], Iteration [30/35], Loss: 0.4816, Cate Loss: 0.0030, Mask Loss: 0.1595
Epoch [26/100] finished, Average Training Loss: 0.8118
Epoch [26/100] finished, Average Validation Loss: 1.3983
Checkpoint saved at ./checkpoints/checkpoint_epoch_26.pth
Epoch [27/100], Iteration [0/35], Loss: 0.8424, Cate Loss: 0.0024, Mask Loss: 0.2800
Epoch [27/100], Iteration [10/35], Loss: 1.3779, Cate Loss: 0.0010, Mask Loss: 0.4590
Epoch [27/100], Iteration [20/35], Loss: 0.2746, Cate Loss: 0.0019, Mask Loss: 0.0909
Epoch [27/100], Iteration [30/35], Loss: 1.5161, Cate Loss: 0.0025, Mask Loss: 0.5045
Epoch [27/100] finished, Average Training Loss: 0.7573
Epoch [27/100] finished, Average Validation Loss: 1.3821
Checkpoint saved at ./checkpoints/checkpoint_epoch_27.pth
Epoch [28/100], Iteration [0/35], Loss: 0.7549, Cate Loss: 0.0025, Mask Loss: 0.2508
Epoch [28/100], Iteration [10/35], Loss: 1.3126, Cate Loss: 0.0010, Mask Loss: 0.4372
Epoch [28/100], Iteration [20/35], Loss: 1.0626, Cate Loss: 0.0017, Mask Loss: 0.3536
Epoch [28/100], Iteration [30/35], Loss: 0.6351, Cate Loss: 0.0031, Mask Loss: 0.2107
Epoch [28/100] finished, Average Training Loss: 0.7721
Epoch [28/100] finished, Average Validation Loss: 1.4418
Checkpoint saved at ./checkpoints/checkpoint_epoch_28.pth
Epoch [29/100], Iteration [0/35], Loss: 0.2635, Cate Loss: 0.0020, Mask Loss: 0.0872
Epoch [29/100], Iteration [10/35], Loss: 0.3043, Cate Loss: 0.0019, Mask Loss: 0.1008
Epoch [29/100], Iteration [20/35], Loss: 0.8067, Cate Loss: 0.0020, Mask Loss: 0.2683
Epoch [29/100], Iteration [30/35], Loss: 1.1972, Cate Loss: 0.0018, Mask Loss: 0.3985
Epoch [29/100] finished, Average Training Loss: 0.7268
Epoch [29/100] finished, Average Validation Loss: 1.4817
Checkpoint saved at ./checkpoints/checkpoint_epoch_29.pth
Epoch [30/100], Iteration [0/35], Loss: 0.5114, Cate Loss: 0.0025, Mask Loss: 0.1696
Epoch [30/100], Iteration [10/35], Loss: 0.5323, Cate Loss: 0.0029, Mask Loss: 0.1765
Epoch [30/100], Iteration [20/35], Loss: 0.6099, Cate Loss: 0.0035, Mask Loss: 0.2021
Epoch [30/100], Iteration [30/35], Loss: 0.9240, Cate Loss: 0.0021, Mask Loss: 0.3073
Epoch [30/100] finished, Average Training Loss: 0.7201
Epoch [30/100] finished, Average Validation Loss: 1.4423
Checkpoint saved at ./checkpoints/checkpoint_epoch_30.pth
Epoch [31/100], Iteration [0/35], Loss: 1.5446, Cate Loss: 0.0025, Mask Loss: 0.5140
Epoch [31/100], Iteration [10/35], Loss: 0.3064, Cate Loss: 0.0024, Mask Loss: 0.1013
Epoch [31/100], Iteration [20/35], Loss: 0.3494, Cate Loss: 0.0018, Mask Loss: 0.1159
Epoch [31/100], Iteration [30/35], Loss: 1.7003, Cate Loss: 0.0025, Mask Loss: 0.5659
Epoch [31/100] finished, Average Training Loss: 0.7303
Epoch [31/100] finished, Average Validation Loss: 1.3427
Checkpoint saved at ./checkpoints/checkpoint_epoch_31.pth
Epoch [32/100], Iteration [0/35], Loss: 0.1598, Cate Loss: 0.0020, Mask Loss: 0.0526
Epoch [32/100], Iteration [10/35], Loss: 1.3036, Cate Loss: 0.0033, Mask Loss: 0.4334
Epoch [32/100], Iteration [20/35], Loss: 0.2687, Cate Loss: 0.0015, Mask Loss: 0.0891
Epoch [32/100], Iteration [30/35], Loss: 0.8653, Cate Loss: 0.0021, Mask Loss: 0.2877
Epoch [32/100] finished, Average Training Loss: 0.6836
Epoch [32/100] finished, Average Validation Loss: 1.3757
Checkpoint saved at ./checkpoints/checkpoint_epoch_32.pth
Epoch [33/100], Iteration [0/35], Loss: 1.4687, Cate Loss: 0.0019, Mask Loss: 0.4889
Epoch [33/100], Iteration [10/35], Loss: 0.1185, Cate Loss: 0.0016, Mask Loss: 0.0390
Epoch [33/100], Iteration [20/35], Loss: 0.6340, Cate Loss: 0.0026, Mask Loss: 0.2104
Epoch [33/100], Iteration [30/35], Loss: 0.6908, Cate Loss: 0.0013, Mask Loss: 0.2298
Epoch [33/100] finished, Average Training Loss: 0.6543
Epoch [33/100] finished, Average Validation Loss: 1.3972
Checkpoint saved at ./checkpoints/checkpoint_epoch_33.pth
Epoch [34/100], Iteration [0/35], Loss: 0.3701, Cate Loss: 0.0015, Mask Loss: 0.1229
Epoch [34/100], Iteration [10/35], Loss: 0.1751, Cate Loss: 0.0015, Mask Loss: 0.0579
Epoch [34/100], Iteration [20/35], Loss: 0.9367, Cate Loss: 0.0016, Mask Loss: 0.3117
Epoch [34/100], Iteration [30/35], Loss: 0.6586, Cate Loss: 0.0037, Mask Loss: 0.2183
Epoch [34/100] finished, Average Training Loss: 0.6161
Epoch [34/100] finished, Average Validation Loss: 1.3701
Checkpoint saved at ./checkpoints/checkpoint_epoch_34.pth
Epoch [35/100], Iteration [0/35], Loss: 0.1390, Cate Loss: 0.0013, Mask Loss: 0.0459
Epoch [35/100], Iteration [10/35], Loss: 0.6672, Cate Loss: 0.0025, Mask Loss: 0.2216
Epoch [35/100], Iteration [20/35], Loss: 2.5360, Cate Loss: 0.0017, Mask Loss: 0.8447
Epoch [35/100], Iteration [30/35], Loss: 0.1725, Cate Loss: 0.0015, Mask Loss: 0.0570
Epoch [35/100] finished, Average Training Loss: 0.6508
Epoch [35/100] finished, Average Validation Loss: 1.4549
Checkpoint saved at ./checkpoints/checkpoint_epoch_35.pth
Epoch [36/100], Iteration [0/35], Loss: 0.2284, Cate Loss: 0.0016, Mask Loss: 0.0756
Epoch [36/100], Iteration [10/35], Loss: 0.7587, Cate Loss: 0.0024, Mask Loss: 0.2521
Epoch [36/100], Iteration [20/35], Loss: 0.2612, Cate Loss: 0.0009, Mask Loss: 0.0868
Epoch [36/100], Iteration [30/35], Loss: 1.2333, Cate Loss: 0.0028, Mask Loss: 0.4102
Epoch [36/100] finished, Average Training Loss: 0.6410
Epoch [36/100] finished, Average Validation Loss: 1.4371
Checkpoint saved at ./checkpoints/checkpoint_epoch_36.pth
Epoch [37/100], Iteration [0/35], Loss: 0.1444, Cate Loss: 0.0012, Mask Loss: 0.0477
Epoch [37/100], Iteration [10/35], Loss: 0.3613, Cate Loss: 0.0023, Mask Loss: 0.1197
Epoch [37/100], Iteration [20/35], Loss: 0.3883, Cate Loss: 0.0014, Mask Loss: 0.1290
Epoch [37/100], Iteration [30/35], Loss: 1.0679, Cate Loss: 0.0014, Mask Loss: 0.3555
Epoch [37/100] finished, Average Training Loss: 0.6118
Epoch [37/100] finished, Average Validation Loss: 1.3472
Checkpoint saved at ./checkpoints/checkpoint_epoch_37.pth
Epoch [38/100], Iteration [0/35], Loss: 0.2426, Cate Loss: 0.0016, Mask Loss: 0.0803
Epoch [38/100], Iteration [10/35], Loss: 0.3964, Cate Loss: 0.0023, Mask Loss: 0.1314
Epoch [38/100], Iteration [20/35], Loss: 0.5591, Cate Loss: 0.0022, Mask Loss: 0.1856
Epoch [38/100], Iteration [30/35], Loss: 0.6829, Cate Loss: 0.0019, Mask Loss: 0.2270
Epoch [38/100] finished, Average Training Loss: 0.5994
Epoch [38/100] finished, Average Validation Loss: 1.3428
Checkpoint saved at ./checkpoints/checkpoint_epoch_38.pth
Epoch [39/100], Iteration [0/35], Loss: 0.8684, Cate Loss: 0.0014, Mask Loss: 0.2890
Epoch [39/100], Iteration [10/35], Loss: 0.4178, Cate Loss: 0.0028, Mask Loss: 0.1383
Epoch [39/100], Iteration [20/35], Loss: 0.3542, Cate Loss: 0.0013, Mask Loss: 0.1177
Epoch [39/100], Iteration [30/35], Loss: 0.6423, Cate Loss: 0.0022, Mask Loss: 0.2134
Epoch [39/100] finished, Average Training Loss: 0.6025
Epoch [39/100] finished, Average Validation Loss: 1.3341
Checkpoint saved at ./checkpoints/checkpoint_epoch_39.pth
Epoch [40/100], Iteration [0/35], Loss: 0.1995, Cate Loss: 0.0016, Mask Loss: 0.0660
Epoch [40/100], Iteration [10/35], Loss: 0.3861, Cate Loss: 0.0010, Mask Loss: 0.1283
Epoch [40/100], Iteration [20/35], Loss: 0.8321, Cate Loss: 0.0039, Mask Loss: 0.2761
Epoch [40/100], Iteration [30/35], Loss: 0.2599, Cate Loss: 0.0014, Mask Loss: 0.0862
Epoch [40/100] finished, Average Training Loss: 0.5911
Epoch [40/100] finished, Average Validation Loss: 1.3723
Checkpoint saved at ./checkpoints/checkpoint_epoch_40.pth
Epoch [41/100], Iteration [0/35], Loss: 0.1740, Cate Loss: 0.0018, Mask Loss: 0.0574
Epoch [41/100], Iteration [10/35], Loss: 1.1388, Cate Loss: 0.0033, Mask Loss: 0.3785
Epoch [41/100], Iteration [20/35], Loss: 0.2231, Cate Loss: 0.0016, Mask Loss: 0.0738
Epoch [41/100], Iteration [30/35], Loss: 0.4090, Cate Loss: 0.0024, Mask Loss: 0.1355
Epoch [41/100] finished, Average Training Loss: 0.5618
Epoch [41/100] finished, Average Validation Loss: 1.3364
Checkpoint saved at ./checkpoints/checkpoint_epoch_41.pth
Epoch [42/100], Iteration [0/35], Loss: 1.0562, Cate Loss: 0.0023, Mask Loss: 0.3513
Epoch [42/100], Iteration [10/35], Loss: 1.1148, Cate Loss: 0.0029, Mask Loss: 0.3706
Epoch [42/100], Iteration [20/35], Loss: 0.5918, Cate Loss: 0.0032, Mask Loss: 0.1962
Epoch [42/100], Iteration [30/35], Loss: 0.1763, Cate Loss: 0.0012, Mask Loss: 0.0584
Epoch [42/100] finished, Average Training Loss: 0.5484
Epoch [42/100] finished, Average Validation Loss: 1.4453
Checkpoint saved at ./checkpoints/checkpoint_epoch_42.pth
Epoch [43/100], Iteration [0/35], Loss: 0.5632, Cate Loss: 0.0022, Mask Loss: 0.1870
Epoch [43/100], Iteration [10/35], Loss: 1.3022, Cate Loss: 0.0016, Mask Loss: 0.4335
Epoch [43/100], Iteration [20/35], Loss: 0.3820, Cate Loss: 0.0013, Mask Loss: 0.1269
Epoch [43/100], Iteration [30/35], Loss: 0.3533, Cate Loss: 0.0024, Mask Loss: 0.1170
Epoch [43/100] finished, Average Training Loss: 0.5461
Epoch [43/100] finished, Average Validation Loss: 1.3199
Checkpoint saved at ./checkpoints/checkpoint_epoch_43.pth
Epoch [44/100], Iteration [0/35], Loss: 0.5422, Cate Loss: 0.0022, Mask Loss: 0.1800
Epoch [44/100], Iteration [10/35], Loss: 1.0925, Cate Loss: 0.0011, Mask Loss: 0.3638
Epoch [44/100], Iteration [20/35], Loss: 0.1553, Cate Loss: 0.0014, Mask Loss: 0.0513
Epoch [44/100], Iteration [30/35], Loss: 0.3656, Cate Loss: 0.0014, Mask Loss: 0.1214
Epoch [44/100] finished, Average Training Loss: 0.5320
Epoch [44/100] finished, Average Validation Loss: 1.3317
Checkpoint saved at ./checkpoints/checkpoint_epoch_44.pth
Epoch [45/100], Iteration [0/35], Loss: 1.1919, Cate Loss: 0.0027, Mask Loss: 0.3964
Epoch [45/100], Iteration [10/35], Loss: 0.3243, Cate Loss: 0.0011, Mask Loss: 0.1077
Epoch [45/100], Iteration [20/35], Loss: 0.8350, Cate Loss: 0.0009, Mask Loss: 0.2780
Epoch [45/100], Iteration [30/35], Loss: 0.1761, Cate Loss: 0.0015, Mask Loss: 0.0582
Epoch [45/100] finished, Average Training Loss: 0.5399
Epoch [45/100] finished, Average Validation Loss: 1.3114
Checkpoint saved at ./checkpoints/checkpoint_epoch_45.pth
Epoch [46/100], Iteration [0/35], Loss: 1.1434, Cate Loss: 0.0031, Mask Loss: 0.3801
Epoch [46/100], Iteration [10/35], Loss: 1.0460, Cate Loss: 0.0008, Mask Loss: 0.3484
Epoch [46/100], Iteration [20/35], Loss: 0.2208, Cate Loss: 0.0014, Mask Loss: 0.0731
Epoch [46/100], Iteration [30/35], Loss: 0.1047, Cate Loss: 0.0013, Mask Loss: 0.0345
Epoch [46/100] finished, Average Training Loss: 0.5124
Epoch [46/100] finished, Average Validation Loss: 1.3505
Checkpoint saved at ./checkpoints/checkpoint_epoch_46.pth
Epoch [47/100], Iteration [0/35], Loss: 0.8223, Cate Loss: 0.0020, Mask Loss: 0.2734
Epoch [47/100], Iteration [10/35], Loss: 0.7299, Cate Loss: 0.0020, Mask Loss: 0.2426
Epoch [47/100], Iteration [20/35], Loss: 0.2340, Cate Loss: 0.0017, Mask Loss: 0.0774
Epoch [47/100], Iteration [30/35], Loss: 0.1375, Cate Loss: 0.0013, Mask Loss: 0.0454
Epoch [47/100] finished, Average Training Loss: 0.5117
Epoch [47/100] finished, Average Validation Loss: 1.3331
Checkpoint saved at ./checkpoints/checkpoint_epoch_47.pth
Epoch [48/100], Iteration [0/35], Loss: 0.3037, Cate Loss: 0.0016, Mask Loss: 0.1007
Epoch [48/100], Iteration [10/35], Loss: 0.5213, Cate Loss: 0.0025, Mask Loss: 0.1729
Epoch [48/100], Iteration [20/35], Loss: 0.2420, Cate Loss: 0.0030, Mask Loss: 0.0797
Epoch [48/100], Iteration [30/35], Loss: 0.5867, Cate Loss: 0.0018, Mask Loss: 0.1950
Epoch [48/100] finished, Average Training Loss: 0.4804
Epoch [48/100] finished, Average Validation Loss: 1.3039
Checkpoint saved at ./checkpoints/checkpoint_epoch_48.pth
Epoch [49/100], Iteration [0/35], Loss: 0.1550, Cate Loss: 0.0009, Mask Loss: 0.0513
Epoch [49/100], Iteration [10/35], Loss: 0.6005, Cate Loss: 0.0010, Mask Loss: 0.1998
Epoch [49/100], Iteration [20/35], Loss: 0.4458, Cate Loss: 0.0024, Mask Loss: 0.1478
Epoch [49/100], Iteration [30/35], Loss: 0.8365, Cate Loss: 0.0021, Mask Loss: 0.2781
Epoch [49/100] finished, Average Training Loss: 0.4923
Epoch [49/100] finished, Average Validation Loss: 1.3387
Checkpoint saved at ./checkpoints/checkpoint_epoch_49.pth
Epoch [50/100], Iteration [0/35], Loss: 0.1237, Cate Loss: 0.0011, Mask Loss: 0.0409
Epoch [50/100], Iteration [10/35], Loss: 0.2781, Cate Loss: 0.0016, Mask Loss: 0.0922
Epoch [50/100], Iteration [20/35], Loss: 0.9150, Cate Loss: 0.0018, Mask Loss: 0.3044
Epoch [50/100], Iteration [30/35], Loss: 0.2542, Cate Loss: 0.0017, Mask Loss: 0.0841
Epoch [50/100] finished, Average Training Loss: 0.4645
Epoch [50/100] finished, Average Validation Loss: 1.3678
Checkpoint saved at ./checkpoints/checkpoint_epoch_50.pth
Epoch [51/100], Iteration [0/35], Loss: 0.5938, Cate Loss: 0.0026, Mask Loss: 0.1970
Epoch [51/100], Iteration [10/35], Loss: 1.0255, Cate Loss: 0.0015, Mask Loss: 0.3413
Epoch [51/100], Iteration [20/35], Loss: 0.5384, Cate Loss: 0.0027, Mask Loss: 0.1785
Epoch [51/100], Iteration [30/35], Loss: 0.6848, Cate Loss: 0.0016, Mask Loss: 0.2277
Epoch [51/100] finished, Average Training Loss: 0.4870
Epoch [51/100] finished, Average Validation Loss: 1.3049
Checkpoint saved at ./checkpoints/checkpoint_epoch_51.pth
Epoch [52/100], Iteration [0/35], Loss: 0.3896, Cate Loss: 0.0012, Mask Loss: 0.1295
Epoch [52/100], Iteration [10/35], Loss: 0.3475, Cate Loss: 0.0023, Mask Loss: 0.1151
Epoch [52/100], Iteration [20/35], Loss: 0.2356, Cate Loss: 0.0014, Mask Loss: 0.0781
Epoch [52/100], Iteration [30/35], Loss: 0.4322, Cate Loss: 0.0008, Mask Loss: 0.1438
Epoch [52/100] finished, Average Training Loss: 0.4684
Epoch [52/100] finished, Average Validation Loss: 1.2826
Checkpoint saved at ./checkpoints/checkpoint_epoch_52.pth
Epoch [53/100], Iteration [0/35], Loss: 0.2574, Cate Loss: 0.0011, Mask Loss: 0.0854
Epoch [53/100], Iteration [10/35], Loss: 0.2584, Cate Loss: 0.0019, Mask Loss: 0.0855
Epoch [53/100], Iteration [20/35], Loss: 1.0530, Cate Loss: 0.0019, Mask Loss: 0.3504
Epoch [53/100], Iteration [30/35], Loss: 1.1157, Cate Loss: 0.0013, Mask Loss: 0.3714
Epoch [53/100] finished, Average Training Loss: 0.4492
Epoch [53/100] finished, Average Validation Loss: 1.3438
Checkpoint saved at ./checkpoints/checkpoint_epoch_53.pth
Epoch [54/100], Iteration [0/35], Loss: 0.9191, Cate Loss: 0.0015, Mask Loss: 0.3059
Epoch [54/100], Iteration [10/35], Loss: 0.9320, Cate Loss: 0.0017, Mask Loss: 0.3101
Epoch [54/100], Iteration [20/35], Loss: 0.2803, Cate Loss: 0.0016, Mask Loss: 0.0929
Epoch [54/100], Iteration [30/35], Loss: 0.4597, Cate Loss: 0.0016, Mask Loss: 0.1527
Epoch [54/100] finished, Average Training Loss: 0.4245
Epoch [54/100] finished, Average Validation Loss: 1.3565
Checkpoint saved at ./checkpoints/checkpoint_epoch_54.pth
Epoch [55/100], Iteration [0/35], Loss: 0.5604, Cate Loss: 0.0019, Mask Loss: 0.1862
Epoch [55/100], Iteration [10/35], Loss: 0.4921, Cate Loss: 0.0018, Mask Loss: 0.1634
Epoch [55/100], Iteration [20/35], Loss: 0.3771, Cate Loss: 0.0033, Mask Loss: 0.1246
Epoch [55/100], Iteration [30/35], Loss: 0.5574, Cate Loss: 0.0028, Mask Loss: 0.1849
Epoch [55/100] finished, Average Training Loss: 0.4237
Epoch [55/100] finished, Average Validation Loss: 1.3095
Checkpoint saved at ./checkpoints/checkpoint_epoch_55.pth
Epoch [56/100], Iteration [0/35], Loss: 0.4726, Cate Loss: 0.0013, Mask Loss: 0.1571
Epoch [56/100], Iteration [10/35], Loss: 0.3018, Cate Loss: 0.0016, Mask Loss: 0.1001
Epoch [56/100], Iteration [20/35], Loss: 0.9353, Cate Loss: 0.0017, Mask Loss: 0.3112
Epoch [56/100], Iteration [30/35], Loss: 0.5852, Cate Loss: 0.0035, Mask Loss: 0.1939
Epoch [56/100] finished, Average Training Loss: 0.4291
Epoch [56/100] finished, Average Validation Loss: 1.3458
Checkpoint saved at ./checkpoints/checkpoint_epoch_56.pth
Epoch [57/100], Iteration [0/35], Loss: 0.3836, Cate Loss: 0.0018, Mask Loss: 0.1273
Epoch [57/100], Iteration [10/35], Loss: 0.5919, Cate Loss: 0.0024, Mask Loss: 0.1965
Epoch [57/100], Iteration [20/35], Loss: 0.1859, Cate Loss: 0.0012, Mask Loss: 0.0616
Epoch [57/100], Iteration [30/35], Loss: 0.0777, Cate Loss: 0.0011, Mask Loss: 0.0255
Epoch [57/100] finished, Average Training Loss: 0.4082
Epoch [57/100] finished, Average Validation Loss: 1.3076
Checkpoint saved at ./checkpoints/checkpoint_epoch_57.pth
Epoch [58/100], Iteration [0/35], Loss: 0.1472, Cate Loss: 0.0010, Mask Loss: 0.0487
Epoch [58/100], Iteration [10/35], Loss: 0.1108, Cate Loss: 0.0014, Mask Loss: 0.0365
Epoch [58/100], Iteration [20/35], Loss: 0.2086, Cate Loss: 0.0010, Mask Loss: 0.0692
Epoch [58/100], Iteration [30/35], Loss: 0.4722, Cate Loss: 0.0030, Mask Loss: 0.1564
Epoch [58/100] finished, Average Training Loss: 0.4015
Epoch [58/100] finished, Average Validation Loss: 1.3008
Checkpoint saved at ./checkpoints/checkpoint_epoch_58.pth
Epoch [59/100], Iteration [0/35], Loss: 0.2389, Cate Loss: 0.0019, Mask Loss: 0.0790
Epoch [59/100], Iteration [10/35], Loss: 0.2004, Cate Loss: 0.0011, Mask Loss: 0.0664
Epoch [59/100], Iteration [20/35], Loss: 0.1237, Cate Loss: 0.0013, Mask Loss: 0.0408
Epoch [59/100], Iteration [30/35], Loss: 0.1084, Cate Loss: 0.0015, Mask Loss: 0.0356
Epoch [59/100] finished, Average Training Loss: 0.4012
Epoch [59/100] finished, Average Validation Loss: 1.3183
Checkpoint saved at ./checkpoints/checkpoint_epoch_59.pth
Epoch [60/100], Iteration [0/35], Loss: 0.1457, Cate Loss: 0.0013, Mask Loss: 0.0481
Epoch [60/100], Iteration [10/35], Loss: 0.1225, Cate Loss: 0.0012, Mask Loss: 0.0404
Epoch [60/100], Iteration [20/35], Loss: 0.5157, Cate Loss: 0.0020, Mask Loss: 0.1713
Epoch [60/100], Iteration [30/35], Loss: 0.4333, Cate Loss: 0.0028, Mask Loss: 0.1435
Epoch [60/100] finished, Average Training Loss: 0.3899
Epoch [60/100] finished, Average Validation Loss: 1.3392
Checkpoint saved at ./checkpoints/checkpoint_epoch_60.pth
Epoch [61/100], Iteration [0/35], Loss: 0.3608, Cate Loss: 0.0012, Mask Loss: 0.1199
Epoch [61/100], Iteration [10/35], Loss: 0.1195, Cate Loss: 0.0012, Mask Loss: 0.0394
Epoch [61/100], Iteration [20/35], Loss: 0.7003, Cate Loss: 0.0011, Mask Loss: 0.2331
Epoch [61/100], Iteration [30/35], Loss: 0.3859, Cate Loss: 0.0015, Mask Loss: 0.1281
Epoch [61/100] finished, Average Training Loss: 0.3811
Epoch [61/100] finished, Average Validation Loss: 1.3355
Checkpoint saved at ./checkpoints/checkpoint_epoch_61.pth
Epoch [62/100], Iteration [0/35], Loss: 0.3529, Cate Loss: 0.0012, Mask Loss: 0.1172
Epoch [62/100], Iteration [10/35], Loss: 0.1134, Cate Loss: 0.0009, Mask Loss: 0.0375
Epoch [62/100], Iteration [20/35], Loss: 0.1343, Cate Loss: 0.0011, Mask Loss: 0.0444
Epoch [62/100], Iteration [30/35], Loss: 0.7498, Cate Loss: 0.0014, Mask Loss: 0.2495
Epoch [62/100] finished, Average Training Loss: 0.3683
Epoch [62/100] finished, Average Validation Loss: 1.3340
Checkpoint saved at ./checkpoints/checkpoint_epoch_62.pth
Epoch [63/100], Iteration [0/35], Loss: 0.3486, Cate Loss: 0.0013, Mask Loss: 0.1158
Epoch [63/100], Iteration [10/35], Loss: 0.6979, Cate Loss: 0.0015, Mask Loss: 0.2321
Epoch [63/100], Iteration [20/35], Loss: 0.1037, Cate Loss: 0.0009, Mask Loss: 0.0343
Epoch [63/100], Iteration [30/35], Loss: 0.6597, Cate Loss: 0.0017, Mask Loss: 0.2193
Epoch [63/100] finished, Average Training Loss: 0.3759
Epoch [63/100] finished, Average Validation Loss: 1.3437
Checkpoint saved at ./checkpoints/checkpoint_epoch_63.pth
Epoch [64/100], Iteration [0/35], Loss: 0.2493, Cate Loss: 0.0015, Mask Loss: 0.0826
Epoch [64/100], Iteration [10/35], Loss: 0.2614, Cate Loss: 0.0023, Mask Loss: 0.0864
Epoch [64/100], Iteration [20/35], Loss: 0.7876, Cate Loss: 0.0005, Mask Loss: 0.2624
Epoch [64/100], Iteration [30/35], Loss: 0.2568, Cate Loss: 0.0008, Mask Loss: 0.0853
Epoch [64/100] finished, Average Training Loss: 0.3679
Epoch [64/100] finished, Average Validation Loss: 1.3193
Checkpoint saved at ./checkpoints/checkpoint_epoch_64.pth
Epoch [65/100], Iteration [0/35], Loss: 0.1290, Cate Loss: 0.0010, Mask Loss: 0.0427
Epoch [65/100], Iteration [10/35], Loss: 0.1046, Cate Loss: 0.0013, Mask Loss: 0.0344
Epoch [65/100], Iteration [20/35], Loss: 0.9587, Cate Loss: 0.0018, Mask Loss: 0.3190
Epoch [65/100], Iteration [30/35], Loss: 0.5244, Cate Loss: 0.0008, Mask Loss: 0.1745
Epoch [65/100] finished, Average Training Loss: 0.3625
Epoch [65/100] finished, Average Validation Loss: 1.3287
Checkpoint saved at ./checkpoints/checkpoint_epoch_65.pth
Epoch [66/100], Iteration [0/35], Loss: 0.4957, Cate Loss: 0.0026, Mask Loss: 0.1644
Epoch [66/100], Iteration [10/35], Loss: 0.4491, Cate Loss: 0.0022, Mask Loss: 0.1490
Epoch [66/100], Iteration [20/35], Loss: 0.8502, Cate Loss: 0.0008, Mask Loss: 0.2831
Epoch [66/100], Iteration [30/35], Loss: 0.0918, Cate Loss: 0.0010, Mask Loss: 0.0303
Epoch [66/100] finished, Average Training Loss: 0.3639
Epoch [66/100] finished, Average Validation Loss: 1.3365
Checkpoint saved at ./checkpoints/checkpoint_epoch_66.pth
Epoch [67/100], Iteration [0/35], Loss: 0.1988, Cate Loss: 0.0018, Mask Loss: 0.0657
Epoch [67/100], Iteration [10/35], Loss: 0.4182, Cate Loss: 0.0009, Mask Loss: 0.1391
Epoch [67/100], Iteration [20/35], Loss: 0.8090, Cate Loss: 0.0023, Mask Loss: 0.2689
Epoch [67/100], Iteration [30/35], Loss: 1.1967, Cate Loss: 0.0023, Mask Loss: 0.3981
Epoch [67/100] finished, Average Training Loss: 0.3360
Epoch [67/100] finished, Average Validation Loss: 1.3054
Checkpoint saved at ./checkpoints/checkpoint_epoch_67.pth
Epoch [68/100], Iteration [0/35], Loss: 0.2531, Cate Loss: 0.0014, Mask Loss: 0.0839
Epoch [68/100], Iteration [10/35], Loss: 0.3762, Cate Loss: 0.0025, Mask Loss: 0.1246
Epoch [68/100], Iteration [20/35], Loss: 0.3754, Cate Loss: 0.0017, Mask Loss: 0.1246
Epoch [68/100], Iteration [30/35], Loss: 0.2304, Cate Loss: 0.0020, Mask Loss: 0.0761
Epoch [68/100] finished, Average Training Loss: 0.3555
Epoch [68/100] finished, Average Validation Loss: 1.3376
Checkpoint saved at ./checkpoints/checkpoint_epoch_68.pth
Epoch [69/100], Iteration [0/35], Loss: 0.5724, Cate Loss: 0.0023, Mask Loss: 0.1900
Epoch [69/100], Iteration [10/35], Loss: 0.3421, Cate Loss: 0.0017, Mask Loss: 0.1135
Epoch [69/100], Iteration [20/35], Loss: 0.6580, Cate Loss: 0.0018, Mask Loss: 0.2188
Epoch [69/100], Iteration [30/35], Loss: 0.0926, Cate Loss: 0.0008, Mask Loss: 0.0306
Epoch [69/100] finished, Average Training Loss: 0.3453
Epoch [69/100] finished, Average Validation Loss: 1.3178
Checkpoint saved at ./checkpoints/checkpoint_epoch_69.pth
Epoch [70/100], Iteration [0/35], Loss: 0.0720, Cate Loss: 0.0014, Mask Loss: 0.0235
Epoch [70/100], Iteration [10/35], Loss: 0.3825, Cate Loss: 0.0015, Mask Loss: 0.1270
Epoch [70/100], Iteration [20/35], Loss: 0.1052, Cate Loss: 0.0010, Mask Loss: 0.0348
Epoch [70/100], Iteration [30/35], Loss: 0.1952, Cate Loss: 0.0022, Mask Loss: 0.0643
Epoch [70/100] finished, Average Training Loss: 0.3385
Epoch [70/100] finished, Average Validation Loss: 1.3532
Checkpoint saved at ./checkpoints/checkpoint_epoch_70.pth
Epoch [71/100], Iteration [0/35], Loss: 0.1151, Cate Loss: 0.0010, Mask Loss: 0.0380
Epoch [71/100], Iteration [10/35], Loss: 0.5841, Cate Loss: 0.0015, Mask Loss: 0.1942
Epoch [71/100], Iteration [20/35], Loss: 0.3732, Cate Loss: 0.0014, Mask Loss: 0.1239
Epoch [71/100], Iteration [30/35], Loss: 0.2230, Cate Loss: 0.0014, Mask Loss: 0.0739
Epoch [71/100] finished, Average Training Loss: 0.3175
Epoch [71/100] finished, Average Validation Loss: 1.3203
Checkpoint saved at ./checkpoints/checkpoint_epoch_71.pth
Epoch [72/100], Iteration [0/35], Loss: 0.2160, Cate Loss: 0.0015, Mask Loss: 0.0715
Epoch [72/100], Iteration [10/35], Loss: 0.4525, Cate Loss: 0.0023, Mask Loss: 0.1500
Epoch [72/100], Iteration [20/35], Loss: 0.1357, Cate Loss: 0.0010, Mask Loss: 0.0449
Epoch [72/100], Iteration [30/35], Loss: 0.0980, Cate Loss: 0.0012, Mask Loss: 0.0323
Epoch [72/100] finished, Average Training Loss: 0.3322
Epoch [72/100] finished, Average Validation Loss: 1.3269
Checkpoint saved at ./checkpoints/checkpoint_epoch_72.pth
Epoch [73/100], Iteration [0/35], Loss: 0.4641, Cate Loss: 0.0020, Mask Loss: 0.1540
Epoch [73/100], Iteration [10/35], Loss: 0.2542, Cate Loss: 0.0013, Mask Loss: 0.0843
Epoch [73/100], Iteration [20/35], Loss: 0.2187, Cate Loss: 0.0015, Mask Loss: 0.0724
Epoch [73/100], Iteration [30/35], Loss: 0.2000, Cate Loss: 0.0007, Mask Loss: 0.0664
Epoch [73/100] finished, Average Training Loss: 0.3153
Epoch [73/100] finished, Average Validation Loss: 1.3269
Checkpoint saved at ./checkpoints/checkpoint_epoch_73.pth
Epoch [74/100], Iteration [0/35], Loss: 0.0851, Cate Loss: 0.0013, Mask Loss: 0.0279
Epoch [74/100], Iteration [10/35], Loss: 0.0773, Cate Loss: 0.0010, Mask Loss: 0.0255
Epoch [74/100], Iteration [20/35], Loss: 0.5153, Cate Loss: 0.0010, Mask Loss: 0.1714
Epoch [74/100], Iteration [30/35], Loss: 0.3155, Cate Loss: 0.0018, Mask Loss: 0.1046
Epoch [74/100] finished, Average Training Loss: 0.3355
Epoch [74/100] finished, Average Validation Loss: 1.3294
Checkpoint saved at ./checkpoints/checkpoint_epoch_74.pth
Epoch [75/100], Iteration [0/35], Loss: 0.1524, Cate Loss: 0.0009, Mask Loss: 0.0505
Epoch [75/100], Iteration [10/35], Loss: 0.2091, Cate Loss: 0.0014, Mask Loss: 0.0693
Epoch [75/100], Iteration [20/35], Loss: 0.2403, Cate Loss: 0.0018, Mask Loss: 0.0795
Epoch [75/100], Iteration [30/35], Loss: 0.3881, Cate Loss: 0.0008, Mask Loss: 0.1291
Epoch [75/100] finished, Average Training Loss: 0.3214
Epoch [75/100] finished, Average Validation Loss: 1.3295
Checkpoint saved at ./checkpoints/checkpoint_epoch_75.pth
Epoch [76/100], Iteration [0/35], Loss: 0.0897, Cate Loss: 0.0008, Mask Loss: 0.0296
Epoch [76/100], Iteration [10/35], Loss: 0.0693, Cate Loss: 0.0009, Mask Loss: 0.0228
Epoch [76/100], Iteration [20/35], Loss: 0.3578, Cate Loss: 0.0020, Mask Loss: 0.1186
Epoch [76/100], Iteration [30/35], Loss: 0.3041, Cate Loss: 0.0018, Mask Loss: 0.1008
Epoch [76/100] finished, Average Training Loss: 0.3364
Epoch [76/100] finished, Average Validation Loss: 1.3329
Checkpoint saved at ./checkpoints/checkpoint_epoch_76.pth
Epoch [77/100], Iteration [0/35], Loss: 0.1837, Cate Loss: 0.0014, Mask Loss: 0.0608
Epoch [77/100], Iteration [10/35], Loss: 0.1343, Cate Loss: 0.0012, Mask Loss: 0.0444
Epoch [77/100], Iteration [20/35], Loss: 0.5065, Cate Loss: 0.0007, Mask Loss: 0.1686
Epoch [77/100], Iteration [30/35], Loss: 0.6120, Cate Loss: 0.0017, Mask Loss: 0.2034
Epoch [77/100] finished, Average Training Loss: 0.3447
Epoch [77/100] finished, Average Validation Loss: 1.3358
Checkpoint saved at ./checkpoints/checkpoint_epoch_77.pth
Epoch [78/100], Iteration [0/35], Loss: 0.7306, Cate Loss: 0.0015, Mask Loss: 0.2430
Epoch [78/100], Iteration [10/35], Loss: 0.0765, Cate Loss: 0.0011, Mask Loss: 0.0251
Epoch [78/100], Iteration [20/35], Loss: 0.1417, Cate Loss: 0.0017, Mask Loss: 0.0467
Epoch [78/100], Iteration [30/35], Loss: 0.4340, Cate Loss: 0.0011, Mask Loss: 0.1443
Epoch [78/100] finished, Average Training Loss: 0.3413
Epoch [78/100] finished, Average Validation Loss: 1.3352
Checkpoint saved at ./checkpoints/checkpoint_epoch_78.pth
Epoch [79/100], Iteration [0/35], Loss: 1.2727, Cate Loss: 0.0010, Mask Loss: 0.4239
Epoch [79/100], Iteration [10/35], Loss: 0.1106, Cate Loss: 0.0011, Mask Loss: 0.0365
Epoch [79/100], Iteration [20/35], Loss: 0.5050, Cate Loss: 0.0015, Mask Loss: 0.1678
Epoch [79/100], Iteration [30/35], Loss: 0.3955, Cate Loss: 0.0008, Mask Loss: 0.1316
Epoch [79/100] finished, Average Training Loss: 0.3226
Epoch [79/100] finished, Average Validation Loss: 1.3365
Checkpoint saved at ./checkpoints/checkpoint_epoch_79.pth
Epoch [80/100], Iteration [0/35], Loss: 0.5268, Cate Loss: 0.0014, Mask Loss: 0.1751
Epoch [80/100], Iteration [10/35], Loss: 0.8522, Cate Loss: 0.0013, Mask Loss: 0.2836
Epoch [80/100], Iteration [20/35], Loss: 0.3099, Cate Loss: 0.0022, Mask Loss: 0.1026
Epoch [80/100], Iteration [30/35], Loss: 0.1577, Cate Loss: 0.0015, Mask Loss: 0.0521
Epoch [80/100] finished, Average Training Loss: 0.3245
Epoch [80/100] finished, Average Validation Loss: 1.3331
Checkpoint saved at ./checkpoints/checkpoint_epoch_80.pth
Epoch [81/100], Iteration [0/35], Loss: 0.8764, Cate Loss: 0.0013, Mask Loss: 0.2917
Epoch [81/100], Iteration [10/35], Loss: 0.4810, Cate Loss: 0.0023, Mask Loss: 0.1596
Epoch [81/100], Iteration [20/35], Loss: 0.1330, Cate Loss: 0.0007, Mask Loss: 0.0441
Epoch [81/100], Iteration [30/35], Loss: 0.9512, Cate Loss: 0.0005, Mask Loss: 0.3169
Epoch [81/100] finished, Average Training Loss: 0.3055
Epoch [81/100] finished, Average Validation Loss: 1.3375
Checkpoint saved at ./checkpoints/checkpoint_epoch_81.pth
Epoch [82/100], Iteration [0/35], Loss: 0.2808, Cate Loss: 0.0013, Mask Loss: 0.0932
Epoch [82/100], Iteration [10/35], Loss: 0.0946, Cate Loss: 0.0010, Mask Loss: 0.0312
Epoch [82/100], Iteration [20/35], Loss: 0.2323, Cate Loss: 0.0009, Mask Loss: 0.0771
Epoch [82/100], Iteration [30/35], Loss: 0.3606, Cate Loss: 0.0021, Mask Loss: 0.1195
Epoch [82/100] finished, Average Training Loss: 0.3102
Epoch [82/100] finished, Average Validation Loss: 1.3364
Checkpoint saved at ./checkpoints/checkpoint_epoch_82.pth
Epoch [83/100], Iteration [0/35], Loss: 0.0477, Cate Loss: 0.0011, Mask Loss: 0.0155
Epoch [83/100], Iteration [10/35], Loss: 0.4335, Cate Loss: 0.0020, Mask Loss: 0.1438
Epoch [83/100], Iteration [20/35], Loss: 0.7817, Cate Loss: 0.0009, Mask Loss: 0.2603
Epoch [83/100], Iteration [30/35], Loss: 0.1612, Cate Loss: 0.0011, Mask Loss: 0.0534
Epoch [83/100] finished, Average Training Loss: 0.3051
Epoch [83/100] finished, Average Validation Loss: 1.3406
Checkpoint saved at ./checkpoints/checkpoint_epoch_83.pth
Epoch [84/100], Iteration [0/35], Loss: 0.5433, Cate Loss: 0.0014, Mask Loss: 0.1806
Epoch [84/100], Iteration [10/35], Loss: 0.0803, Cate Loss: 0.0011, Mask Loss: 0.0264
Epoch [84/100], Iteration [20/35], Loss: 0.3102, Cate Loss: 0.0016, Mask Loss: 0.1029
Epoch [84/100], Iteration [30/35], Loss: 0.1373, Cate Loss: 0.0009, Mask Loss: 0.0455
Epoch [84/100] finished, Average Training Loss: 0.3146
Epoch [84/100] finished, Average Validation Loss: 1.3387
Checkpoint saved at ./checkpoints/checkpoint_epoch_84.pth
Epoch [85/100], Iteration [0/35], Loss: 0.0715, Cate Loss: 0.0011, Mask Loss: 0.0235
Epoch [85/100], Iteration [10/35], Loss: 0.3630, Cate Loss: 0.0016, Mask Loss: 0.1205
Epoch [85/100], Iteration [20/35], Loss: 0.1605, Cate Loss: 0.0011, Mask Loss: 0.0531
Epoch [85/100], Iteration [30/35], Loss: 0.1037, Cate Loss: 0.0010, Mask Loss: 0.0343
Epoch [85/100] finished, Average Training Loss: 0.3093
Epoch [85/100] finished, Average Validation Loss: 1.3403
Checkpoint saved at ./checkpoints/checkpoint_epoch_85.pth
Epoch [86/100], Iteration [0/35], Loss: 1.7265, Cate Loss: 0.0009, Mask Loss: 0.5752
Epoch [86/100], Iteration [10/35], Loss: 0.0978, Cate Loss: 0.0009, Mask Loss: 0.0323
Epoch [86/100], Iteration [20/35], Loss: 0.1638, Cate Loss: 0.0012, Mask Loss: 0.0542
Epoch [86/100], Iteration [30/35], Loss: 0.1292, Cate Loss: 0.0013, Mask Loss: 0.0426
Epoch [86/100] finished, Average Training Loss: 0.3402
Epoch [86/100] finished, Average Validation Loss: 1.3419
Checkpoint saved at ./checkpoints/checkpoint_epoch_86.pth
Epoch [87/100], Iteration [0/35], Loss: 0.5147, Cate Loss: 0.0019, Mask Loss: 0.1710
Epoch [87/100], Iteration [10/35], Loss: 0.2636, Cate Loss: 0.0016, Mask Loss: 0.0873
Epoch [87/100], Iteration [20/35], Loss: 0.1351, Cate Loss: 0.0011, Mask Loss: 0.0447
Epoch [87/100], Iteration [30/35], Loss: 0.3316, Cate Loss: 0.0028, Mask Loss: 0.1096
Epoch [87/100] finished, Average Training Loss: 0.3188
Epoch [87/100] finished, Average Validation Loss: 1.3423
Checkpoint saved at ./checkpoints/checkpoint_epoch_87.pth
Epoch [88/100], Iteration [0/35], Loss: 0.9807, Cate Loss: 0.0016, Mask Loss: 0.3263
Epoch [88/100], Iteration [10/35], Loss: 0.1117, Cate Loss: 0.0012, Mask Loss: 0.0368
Epoch [88/100], Iteration [20/35], Loss: 0.2477, Cate Loss: 0.0011, Mask Loss: 0.0822
Epoch [88/100], Iteration [30/35], Loss: 0.1248, Cate Loss: 0.0009, Mask Loss: 0.0413
Epoch [88/100] finished, Average Training Loss: 0.3085
Epoch [88/100] finished, Average Validation Loss: 1.3427
Checkpoint saved at ./checkpoints/checkpoint_epoch_88.pth
Epoch [89/100], Iteration [0/35], Loss: 0.3303, Cate Loss: 0.0016, Mask Loss: 0.1096
Epoch [89/100], Iteration [10/35], Loss: 0.1733, Cate Loss: 0.0016, Mask Loss: 0.0572
Epoch [89/100], Iteration [20/35], Loss: 0.0952, Cate Loss: 0.0010, Mask Loss: 0.0314
Epoch [89/100], Iteration [30/35], Loss: 0.1649, Cate Loss: 0.0012, Mask Loss: 0.0546
Epoch [89/100] finished, Average Training Loss: 0.3209
Epoch [89/100] finished, Average Validation Loss: 1.3438
Checkpoint saved at ./checkpoints/checkpoint_epoch_89.pth
Epoch [90/100], Iteration [0/35], Loss: 0.1005, Cate Loss: 0.0010, Mask Loss: 0.0331
Epoch [90/100], Iteration [10/35], Loss: 0.0933, Cate Loss: 0.0010, Mask Loss: 0.0308
Epoch [90/100], Iteration [20/35], Loss: 0.3939, Cate Loss: 0.0015, Mask Loss: 0.1308
Epoch [90/100], Iteration [30/35], Loss: 0.4832, Cate Loss: 0.0022, Mask Loss: 0.1603
Epoch [90/100] finished, Average Training Loss: 0.3107
Epoch [90/100] finished, Average Validation Loss: 1.3456
Checkpoint saved at ./checkpoints/checkpoint_epoch_90.pth
Epoch [91/100], Iteration [0/35], Loss: 1.0804, Cate Loss: 0.0014, Mask Loss: 0.3597
Epoch [91/100], Iteration [10/35], Loss: 0.0856, Cate Loss: 0.0010, Mask Loss: 0.0282
Epoch [91/100], Iteration [20/35], Loss: 0.3134, Cate Loss: 0.0015, Mask Loss: 0.1040
Epoch [91/100], Iteration [30/35], Loss: 0.1614, Cate Loss: 0.0010, Mask Loss: 0.0535
Epoch [91/100] finished, Average Training Loss: 0.3092
Epoch [91/100] finished, Average Validation Loss: 1.3450
Checkpoint saved at ./checkpoints/checkpoint_epoch_91.pth
Epoch [92/100], Iteration [0/35], Loss: 0.6155, Cate Loss: 0.0020, Mask Loss: 0.2045
Epoch [92/100], Iteration [10/35], Loss: 0.2981, Cate Loss: 0.0017, Mask Loss: 0.0988
Epoch [92/100], Iteration [20/35], Loss: 0.0754, Cate Loss: 0.0009, Mask Loss: 0.0248
Epoch [92/100], Iteration [30/35], Loss: 0.0836, Cate Loss: 0.0011, Mask Loss: 0.0275
Epoch [92/100] finished, Average Training Loss: 0.3057
Epoch [92/100] finished, Average Validation Loss: 1.3448
Checkpoint saved at ./checkpoints/checkpoint_epoch_92.pth
Epoch [93/100], Iteration [0/35], Loss: 1.0641, Cate Loss: 0.0021, Mask Loss: 0.3540
Epoch [93/100], Iteration [10/35], Loss: 0.4604, Cate Loss: 0.0006, Mask Loss: 0.1533
Epoch [93/100], Iteration [20/35], Loss: 0.0719, Cate Loss: 0.0008, Mask Loss: 0.0237
Epoch [93/100], Iteration [30/35], Loss: 0.1479, Cate Loss: 0.0008, Mask Loss: 0.0490
Epoch [93/100] finished, Average Training Loss: 0.3107
Epoch [93/100] finished, Average Validation Loss: 1.3447
Checkpoint saved at ./checkpoints/checkpoint_epoch_93.pth
Epoch [94/100], Iteration [0/35], Loss: 0.1178, Cate Loss: 0.0009, Mask Loss: 0.0390
Epoch [94/100], Iteration [10/35], Loss: 0.1274, Cate Loss: 0.0011, Mask Loss: 0.0421
Epoch [94/100], Iteration [20/35], Loss: 0.6710, Cate Loss: 0.0015, Mask Loss: 0.2232
Epoch [94/100], Iteration [30/35], Loss: 0.4614, Cate Loss: 0.0017, Mask Loss: 0.1532
Epoch [94/100] finished, Average Training Loss: 0.3202
Epoch [94/100] finished, Average Validation Loss: 1.3447
Checkpoint saved at ./checkpoints/checkpoint_epoch_94.pth
Epoch [95/100], Iteration [0/35], Loss: 0.1851, Cate Loss: 0.0014, Mask Loss: 0.0612
Epoch [95/100], Iteration [10/35], Loss: 0.2056, Cate Loss: 0.0014, Mask Loss: 0.0680
Epoch [95/100], Iteration [20/35], Loss: 1.3514, Cate Loss: 0.0006, Mask Loss: 0.4503
Epoch [95/100], Iteration [30/35], Loss: 0.1615, Cate Loss: 0.0021, Mask Loss: 0.0532
Epoch [95/100] finished, Average Training Loss: 0.3216
Epoch [95/100] finished, Average Validation Loss: 1.3448
Checkpoint saved at ./checkpoints/checkpoint_epoch_95.pth
Epoch [96/100], Iteration [0/35], Loss: 0.1953, Cate Loss: 0.0015, Mask Loss: 0.0646
Epoch [96/100], Iteration [10/35], Loss: 0.1205, Cate Loss: 0.0010, Mask Loss: 0.0398
Epoch [96/100], Iteration [20/35], Loss: 0.3281, Cate Loss: 0.0015, Mask Loss: 0.1088
Epoch [96/100], Iteration [30/35], Loss: 0.4939, Cate Loss: 0.0020, Mask Loss: 0.1640
Epoch [96/100] finished, Average Training Loss: 0.3254
Epoch [96/100] finished, Average Validation Loss: 1.3449
Checkpoint saved at ./checkpoints/checkpoint_epoch_96.pth
Epoch [97/100], Iteration [0/35], Loss: 0.3213, Cate Loss: 0.0020, Mask Loss: 0.1064
Epoch [97/100], Iteration [10/35], Loss: 0.8440, Cate Loss: 0.0011, Mask Loss: 0.2810
Epoch [97/100], Iteration [20/35], Loss: 0.8770, Cate Loss: 0.0015, Mask Loss: 0.2918
Epoch [97/100], Iteration [30/35], Loss: 0.0986, Cate Loss: 0.0010, Mask Loss: 0.0325
Epoch [97/100] finished, Average Training Loss: 0.3036
Epoch [97/100] finished, Average Validation Loss: 1.3448
Checkpoint saved at ./checkpoints/checkpoint_epoch_97.pth
Epoch [98/100], Iteration [0/35], Loss: 0.0617, Cate Loss: 0.0011, Mask Loss: 0.0202
Epoch [98/100], Iteration [10/35], Loss: 0.1938, Cate Loss: 0.0013, Mask Loss: 0.0642
Epoch [98/100], Iteration [20/35], Loss: 0.4877, Cate Loss: 0.0025, Mask Loss: 0.1618
Epoch [98/100], Iteration [30/35], Loss: 0.1183, Cate Loss: 0.0009, Mask Loss: 0.0392
Epoch [98/100] finished, Average Training Loss: 0.3210
Epoch [98/100] finished, Average Validation Loss: 1.3448
Checkpoint saved at ./checkpoints/checkpoint_epoch_98.pth
Epoch [99/100], Iteration [0/35], Loss: 0.1765, Cate Loss: 0.0015, Mask Loss: 0.0584
Epoch [99/100], Iteration [10/35], Loss: 1.1758, Cate Loss: 0.0012, Mask Loss: 0.3915
Epoch [99/100], Iteration [20/35], Loss: 0.4074, Cate Loss: 0.0008, Mask Loss: 0.1355
Epoch [99/100], Iteration [30/35], Loss: 0.1003, Cate Loss: 0.0009, Mask Loss: 0.0331
Epoch [99/100] finished, Average Training Loss: 0.3091
Epoch [99/100] finished, Average Validation Loss: 1.3450
Checkpoint saved at ./checkpoints/checkpoint_epoch_99.pth
Epoch [100/100], Iteration [0/35], Loss: 0.2151, Cate Loss: 0.0013, Mask Loss: 0.0713
Epoch [100/100], Iteration [10/35], Loss: 0.4479, Cate Loss: 0.0016, Mask Loss: 0.1487
Epoch [100/100], Iteration [20/35], Loss: 0.0866, Cate Loss: 0.0009, Mask Loss: 0.0286
Epoch [100/100], Iteration [30/35], Loss: 0.1612, Cate Loss: 0.0010, Mask Loss: 0.0534
Epoch [100/100] finished, Average Training Loss: 0.3131
Epoch [100/100] finished, Average Validation Loss: 1.3451
Checkpoint saved at ./checkpoints/checkpoint_epoch_100.pth
Training complete.
"""

# Initialize lists to store the losses
epochs = []
total_losses = []
cate_losses = []
mask_losses = []

# Variables to accumulate losses per epoch
epoch_cate_losses = []
epoch_mask_losses = []
epoch_total_losses = []

# Process logs line by line
for line in logs.strip().split('\n'):
    # Match lines with iteration losses
    iter_match = re.match(r'Epoch \[(\d+)/\d+\], Iteration \[\d+/\d+\], Loss: ([\d\.]+), Cate Loss: ([\d\.]+), Mask Loss: ([\d\.]+)', line)
    if iter_match:
        epoch = int(iter_match.group(1))
        total_loss = float(iter_match.group(2))
        cate_loss = float(iter_match.group(3))
        mask_loss = float(iter_match.group(4))
        
        epoch_total_losses.append(total_loss)
        epoch_cate_losses.append(cate_loss)
        epoch_mask_losses.append(mask_loss)
    
    # Match lines indicating the end of an epoch
    epoch_end_match = re.match(r'Epoch \[(\d+)/\d+\] finished, Average Training Loss: ([\d\.]+)', line)
    if epoch_end_match:
        epoch = int(epoch_end_match.group(1))
        avg_total_loss = float(epoch_end_match.group(2))
        
        # Compute average cate_loss and mask_loss for the epoch
        avg_cate_loss = sum(epoch_cate_losses) / len(epoch_cate_losses) if epoch_cate_losses else 0
        avg_mask_loss = sum(epoch_mask_losses) / len(epoch_mask_losses) if epoch_mask_losses else 0
        
        # Append to the lists
        epochs.append(epoch)
        total_losses.append(avg_total_loss)
        cate_losses.append(avg_cate_loss)
        mask_losses.append(avg_mask_loss)
        
        # Reset epoch losses
        epoch_cate_losses = []
        epoch_mask_losses = []
        epoch_total_losses = []

# Plot the losses
plt.figure(figsize=(12, 6))
plt.plot(epochs, total_losses, label='Total Loss')
plt.plot(epochs, cate_losses, label='Focal Loss (Cate Loss)')
plt.plot(epochs, mask_losses, label='Dice Loss (Mask Loss)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.grid(True)
plt.show()
