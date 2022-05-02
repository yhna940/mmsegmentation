import argparse
import os
import random
import re
import shutil
from glob import glob
from os import path as osp
from typing import List

import mmcv
import numpy as np
import ray
from PIL import Image
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Inria dataset')
    parser.add_argument('data_path', help='data path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def set_env(nproc: int):
    ray.init(num_cpus=nproc)
    mmcv.mkdir_or_exist('val')
    mmcv.mkdir_or_exist('val/images')
    mmcv.mkdir_or_exist('val/gt')


@ray.remote
def mv_val(fname: str):
    if not fname.endswith('.tif'):
        return
    idx = re.findall(r'\d+', fname)
    assert len(idx) == 1
    if 1 <= int(idx[0]) < 6:
        shutil.move(
            osp.join('train', 'images', fname),
            osp.join('val', 'images', fname))
        shutil.move(
            osp.join('train', 'gt', fname), osp.join('val', 'gt', fname))
        print(f'MV {fname}')


@ray.remote
def slide_w_cvt(fname: str,
                sub_dirs: List[str],
                crop_size: int = 1000,
                stride: int = 250):

    def cvt_pil(img: np.ndarray, palette: List = [[0, 0, 0], [255, 255, 255]]):
        pil_img = Image.fromarray(img / 255).convert('P')
        pil_img.putpalette(np.array(palette, dtype=np.uint8))
        return pil_img

    if not fname.endswith('.tif'):
        return
    img = io.imread(osp.join(*sub_dirs, fname))
    row_pix = img.shape[0]
    col_pix = img.shape[1]
    for r_idx in range(max(row_pix - crop_size + stride - 1, 0) // stride + 1):
        for c_idx in range(
                max(col_pix - crop_size + stride - 1, 0) // stride + 1):
            rstart = r_idx * stride
            cstart = c_idx * stride
            rend = min(rstart + crop_size, row_pix)
            cend = min(cstart + crop_size, col_pix)
            crop_img = img[rstart:rend, cstart:cend].copy()
            prefix = f'u{crop_size}s{stride}r{r_idx}c{c_idx}_'
            if 'gt' in sub_dirs:
                pil_img = cvt_pil(crop_img)
                save_name = prefix + fname.replace('tif', 'png')
                pil_img.save(osp.join(*sub_dirs, save_name))
            else:
                save_name = prefix + fname.replace('tif', 'jpg')
                io.imsave(
                    osp.join(*sub_dirs, save_name),
                    crop_img,
                    check_contrast=False)
            print(f'SV {save_name}')


def main():
    args = parse_args()
    futs = []
    os.chdir(args.data_path)
    set_env(args.nproc)
    fnames = os.listdir(osp.join('train', 'images'))
    random.shuffle(fnames)
    for fname in fnames:
        futs.append(mv_val.remote(fname))
    ray.get(futs)

    futs = []
    for path in glob('**/*.tif', recursive=True):
        if 'test' in path:
            continue
        sub_dirs = path.split('/')[:-1]
        fname = path.split('/')[-1]
        futs.append(slide_w_cvt.remote(fname, sub_dirs))
    ray.get(futs)


if __name__ == '__main__':
    main()
