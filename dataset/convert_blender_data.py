import json
import os
import argparse
from os import path
import numpy as np
from PIL import Image

def load_renderings(data_dir, split):

    f = "transforms_{}.json".format(split)
    with open(path.join(data_dir, f), "r") as fp:
        mata = json.load(fp)

    imgs = []
    cams = []
    print("loading imgs")
    for frame in meta["frames"]:
        fname = os.path.join(data_dir, frame['file_path'] + ".png")
        with open(fname, "rb") as imgin:
            img = np.array(Image.open(imgin), dtype = np.float32)/255.
        cams.append(frame["transform_matrix"])
        imgs.append(img)

    ret = {}
    ret["image"] = np.stack(imgs, axis = 0)
    print("loaded all imgs, shape is", ret["image"].shape)
    ret["cantoworlds"] = np.stack(cams, axis = 0)
    w = ret["image"].shape[2]
    camera_angle_x = float(meta["camera_angel_x"])
    ret["focal"] = .5 * w / np.tan(.5 * camera_angle_x)

    return ret

def down2(img):

    sh = img.shape
    return np.mean(np.reshape(img, [sh[0]//2, 2, sh[1]//2 ,2,-1]), (1,3))

def convert_to_nerfdata(basedir, newdir, n_down):
    if not os.path.exists(newdir):
        os.mkdirs(newdir)

    splits = ["train", "val", "test"]
    bigmeta = {}

    for split in splits:
        print("Split", split)

        data = load_renderings(basedir, split)

        imgdir = "images_{}".format(split)
        os.makedirs(os.path.join(newdir, imgdir), exist_ok = True)
        fnames = []
        widths = []
        heights = []
        focals = []
        cam2worlds = []
        lossmults = []
        labels = []
        nears, fars = [],[]
        f = data["focal"]
        print("Saving imgages")
        for i, img in enumerate(data["image"]):
            for j in range(n_down):
                fname = "{}/{:03d}_d{}.png".format(imgdir, i, j)
                fnames.append(fname)
                fname = os.path.join(newdir, fname)
                with open(fname, "wb") as imgout:
                    img8 = Image.fromarray(np.uint8(img * 255))
                    img8.save(imgout)

                witdths.append(img.shape[1])
                heights.append(img.shape[0])
                focals.append(f / 2** j)
                cam2worlds.append(data["camtoworlds"][i].tolist())
                lossmults.append(4.** j)
                labels.append(j)
                nears.append(2.)
                fars.append(6.)
                img = down2(img)


        # Create metadata
        meta = {}
        meta['file_path'] = fnames
        meta['cam2world'] = cam2worlds
        meta['width'] = widths
        meta['height'] = heights
        meta['focal'] = focals  ###n
        meta['label'] = labels
        meta['near'] = nears
        meta['far'] = fars
        meta['lossmult'] = lossmults

        fx = np.array(focals)
        fy = np.array(focals)
        cx = np.array(meta['width']) * .5
        cy = np.array(meta['height']) * .5
        arr0 = np.zeros_like(cx)
        arr1 = np.ones_like(cy)

        k_inv = np.array([
            [arr1/fx, arr0,   -cx / fx],
            [arr0,    arr1/fy,-cy / fy],
            [arr0, arr0, arr1]
        ])          ## 3 3 n

        k_inv = np.moveaxis(k_inv, -1, 0)  ###n 3 3
        meta['pix2cam'] = k_inv.tolist()

        bigmeta[split] = meta

    for k in bigmeta:
        for j in bigmeta[k]:
            print(k, j, type(bigmeta[k][j]), np.array(bigmeta[k][j]).shape)

    jsonfile = os.path.join(newdir, 'metadata.json')
    with open(jsonfile, 'w') as f:
        json.dump(bigmeta, f, ensure_ascii=False, indent=4)











