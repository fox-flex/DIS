import os
import shutil
from pathlib import Path
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

from config import hypar, hypar_set_mode, dataset_uhrsd_te, dataset_synt_te, dataset_synt_tr, dataset_duts_te, dataset_duts_tr
import pandas as pd

if __name__ == "__main__":
    datasets = [
        dataset_synt_tr,
        dataset_duts_tr,
        dataset_uhrsd_te,
    ]
    line_ts = ['-','-.','--']

    vis_per_ds = 20
    results = {}

    for dataset, line_t in zip(datasets, line_ts):
        results[dataset["name"]] = [[], [], [], []]
        dataset_path = dataset["im_dir"]
        sizes = []

        im_list = glob(f'{dataset["im_dir"]}{os.sep}*{dataset["im_ext"]}')
        gt_list = glob(f'{dataset["gt_dir"]}{os.sep}*{dataset["gt_ext"]}')
        im_list.sort()
        gt_list.sort()
        to_vis = np.random.randint(0, high=len(im_list), size=vis_per_ds)
        

        assert len(im_list) == len(gt_list), f"Number of images and masks do not match for {dataset['name']}"
        data = []
        n = len(im_list)
        cntr = 5000

        sample = np.zeros((cntr*2, cntr*2), dtype=np.float64)
        for i in tqdm(range(n), total=n):
            im_path, gt_path = im_list[i], gt_list[i]

            # im = io.imread(im_path)
            gt = io.imread(gt_path)

            if len(gt.shape) > 2:
                gt = gt.mean(axis=2)
            
            size = gt.shape[0] * gt.shape[1]
            
            sizes.append(np.sum(gt>0) / size)
            continue
            
            h, w = gt.shape
            y = cntr - h // 2
            x = cntr - w // 2
            sample[y:y+h, x:x+w] += gt>0
        
        import matplotlib.pyplot as plt
        import scipy.stats as stats

        sizes = np.array(sizes)
        # Build histogram
        w = (np.zeros_like(sizes) + 1. / sizes.size * 100)
        plt.hist(sizes, bins=20, range=(0, 1), weights=w,
            histtype=u'step',
            label=dataset["name"].split("_")[0],
            linestyle=line_t,
            linewidth=2)
        plt.legend()
        plt.title('Distribution of salient object sizes')
        plt.xlabel('Salient region size')
        plt.ylabel('Percent of images')
        # Scale y axis in percents
        # plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

        # Save plot to file'.png'
        plt.savefig('hist.jpg')
        
        # B = np.argwhere(sample>0)
        # (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1 
        # to_el = max(cntr - ystart, cntr - xstart, ystop - cntr, xstop - cntr)
        # sample = sample[cntr-to_el:cntr+to_el, cntr-to_el:cntr+to_el]
        # sample /= sample.max()
        # io.imsave(f'{dataset["name"]}.jpg', sample)
