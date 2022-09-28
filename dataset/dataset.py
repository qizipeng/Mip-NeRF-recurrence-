import os
import json
import collections
from torch.utils.data import Dataset
import numpy as np


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far')
)
Rays_keys = Rays._fields




class BaseDataset(Dataset):
    """ Base dataset   """

    def __init__(self, data_dir, split, white_bkgd = Ture, batch_type = "all_imgs", factor = 0):
        super(BaseDataset,self).__init__()

        self.near = 2
        self.far = 6
        self.split = split
        self.data_dir = data_dir
        self.white_bkgd = white_bkgd
        self.batch_type = batch_type
        self.images = None
        self.rays = None
        self.it = -1
        self.n_example = 1
        self.factor = factor

    def _flatten(self,x):
         x = [y.reshape(-1,y.shape(-1)) for y in x]
         if self.batch_type == "all_imgs":
             x = np.concatenate(x, axis = 0)

         return x

    def _train_init(self):

        self._load_renderings()
        self._generate_rays()

        if self.split == "train":
            assert self.batch_type == "all_imgs" "the batch_type can only be all_imgs with flatten"

            self.images = self._flatten(self.images)
            self.rays = namedtuple_map(self.rays)

        else:
            assert self.batch_type == "single_img" "the batch_type can only be single_img without flatten"


    def _val_init(self):

        self._load_renderings()
        self._generate_rays()

    def _generate_rays(self):

        raise ValueError("Implement in different dataset")

    def _load_renderings(self):

        raise ValueError("Implement in different dataset")

    def __len__(self):

        return len(self.images)

    def __getitem__(self, item):

        if self.split == "val":
            index = (self.it + 1) % self.n_example
            self.it +=1

        rays = Rays(*[getattr(self.rays, key)[index] for key in Rays_key])
        return rays, self.images[index]

class Multicam(BaseDataset):

    def __inti__(self, data_dir, split = "train", white_bkgd = True, batch_type = "all_imgs"):
        super(Multicam,self).__init__(data_dir, split, white_bkgd, batch_type)

        if self.split == "train":
            self._train_init()
        else:
            assert batch_type == "single_img" "The batch_type can only be single_img without flatten"

            self._val_init()

    def _load_renderings(self):

        with open(os.path.join(self.data_dir, "metadata.json"), 'r') as fp:
            self.meta = josn.load(fp)[self.split]


        slef.meta = {k: np.array(self.meat[k]) for k in self.meta}


        imgs = []
        for relative_path in self.meta["file_path"]:
            img_path = os.path.join(self.data_dir,relative_path)
            with open(img_path,"rb") as img_file:
                img = np.array(Image.open(img_file), dtype = np.float32)/255.
                if self.white_bkgd:
                    img = img[...,:3] * img[...,-1:] + (1. - img[...,-1:])

                imgs .appen(img)

        self.imges = imgs
        del imgs
        self.n_example = len(self.images)

    def _generate_rays(self):

        pix2cam = self.meta["pix2cam"].astype(np.float32)  ###内参
        cam2world  = self.meta["cam2world"].astype(np.float32) ###外参
        width = self.meta["width"].astype(np.float32)
        height = self.meta["height"].astype(np.float32)

        def res2grid(w, h):
            return np.meshgrid(
                np.arange(w, dtype = np.float32) +.5,
                np.arange(h, dtype = np.float32) + .5,
                indexing = "xy"
            )

        xy = [res2grid(w, d) for w, d in zip(width, height)]
        pixel_dirs = [np.stack([x,y,np.ones_like(x)], axis = 1) for x,y in xy]  #n个 h w 1
        camera_dirs = [v @ p2c[:3,:3].T for v, p2c in zip(pixel_dirs, pix2cam)]  ## x y 1  pix坐标系->cam坐标系
        directions = [(v @ c2w[:3,:3].T).copy() for v, c2w in zip(camera_dirs, cam2world)] ##cam坐标系->世界坐标系 n个相机视角 x y z
        origins = [
            np.broadcast_to(c2w[:3, -1], v.shape).copy()
            for v, c2w in zip(directions, cam2world)
        ]  ####相机外参的最后一列是相机的世界坐标  广播 将【1，3】->【n，3】 相机的世界坐标


        ###做归一化
        viewdirs = [
            v/np.linalog.norm(v, axis = -1, keepdims = True) for v in directions
        ]

        def broadcast_scalar_attribute(x):
            return[
                np.broadcast_to(x[i], origins[i][...,:1].shape).astype(np.float32)
                for i in range(len(self.images))
            ]

        lossmult = broadcast_scalar_attribute(self.meta["lossmult"].copy())
        near = broadcast_scalar_attribute(self.meta["near"].copy())
        far = broadcast_scalar_attribute(self.meta["far"].copy())


        ##v[:-1 - v[1: 相邻向量在x分量 x y
        dx = [
            np.sqrt(np.sum((v[:-1,:,:]-v[1:,:,:])**2,-1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1,:],0]) for v in dx]
        radii = [v[...,None]*2 / np.sqrt(12) for v in dx]   ###这里是圆的半径吧 在世界坐标系下的


























