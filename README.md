Dataset image examples in [dataset_imgs](dataset_imgs).

FPN image examples in [fpn_imgs](fpn_imgs).
This difference between an object different levels of the FPN feature pyramid is different resolutions, since the levels separate objects by size (in overlapping intervals), as obtained by different-stride convolutions at each level. The lower levels have higher resolution lower semantic info, higher levels are the opposite. 

test `solo_head.py` and `dataset.py` by running the `main.py` script. 

colab training: https://colab.research.google.com/drive/11hfi9txNYtI5g8juv4Ef7zOIP82FJ3MT?usp=sharing

can just run main_infer.py in vscode