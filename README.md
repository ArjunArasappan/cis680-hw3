FPN image examples in [fpn_imgs](fpn_imgs).
This difference between an object different levels of the FPN feature pyramid is different resolutions, since the levels separate objects by size (in overlapping intervals), as obtained by different-stride convolutions at each level. The lower levels have higher resolution lower semantic info, higher levels are the opposite. 

test `solo_head.py` by just running it with `__main__`