'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 01:02:36
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-13 15:38:11
'''
from itertools import product

import torch
from math import sqrt


class Priors:
    def __init__(self,clip = True):
        self.image_size = 512
        self.strides = [4,4,8,8,16,32,32,32]
        self.feature_maps = [127,127,63,63,31,15,15,15]
        self.sizes = [55,71,111,143,223,383,511,639]
        self.clip = clip
    def __call__(self):
        priors = []
        for k, f in enumerate(self.feature_maps):
            # 513/4 = 128.25
            # 126/128.25 = 0.98245
            # 126.5/128.25 = 0.98635
            scale = self.image_size / self.strides[k]
            for i, j in product(range(f), repeat=2):

                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale
                #r = self.sizes[k]
                #r = r/self.image_size
                h = w = self.sizes[k] / self.image_size
                priors.append([cx, cy,w,h])

        priors = torch.tensor(priors) #(num_priors,4)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
def get_priors():
    return Priors()
if __name__ == "__main__":
    priors = Priors()
    print(priors().size())
