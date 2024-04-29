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

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *
from config import hypar, hypar_set_mode, dataset_uhrsd_te, dataset_synt_te, dataset_duts_te
import pandas as pd

def get_IoU(pred, gt):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_pr_rc_f1(pred, gt):
    tp = np.sum(np.logical_and(pred == 1, gt == 1))
    fp = np.sum(np.logical_and(pred == 1, gt == 0))
    fn = np.sum(np.logical_and(pred == 0, gt == 1))
    tn = np.sum(np.logical_and(pred == 0, gt == 0))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

if __name__ == "__main__":
    datasets = [
        dataset_uhrsd_te,
        dataset_synt_te,
        dataset_duts_te,
    ]
    
    model_path = ''
    result_path = Path("../demo_dataset/comb-3")  #The folder path that you want to save the results


    input_size=[1024,1024]
    net=ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net=net.cuda()
    else:
        net.load_state_dict(torch.load(model_path,map_location="cpu"))
    net.eval()
    vis_per_ds = 10
    results = {}
    with torch.no_grad():

        for dataset in datasets:
            results[dataset["name"]] = [[], [], [], []]
            dataset_path = dataset["im_dir"]
            dataset_vis = result_path / dataset["name"]

            if os.path.exists(dataset_vis):
                shutil.rmtree(dataset_vis)
            os.makedirs(dataset_vis)
            im_list = glob(f'{dataset["im_dir"]}{os.sep}*{dataset["im_ext"]}')
            gt_list = glob(f'{dataset["gt_dir"]}{os.sep}*{dataset["gt_ext"]}')
            im_list.sort()
            gt_list.sort()
            to_vis = set(np.random.randint(0, high=len(im_list), size=vis_per_ds))

            assert len(im_list) == len(gt_list), f"Number of images and masks do not match for {dataset['name']}"
            with tqdm(enumerate(zip(im_list, gt_list)), total=len(im_list)) as pbar:
                for i, (im_path, gt_path) in pbar:
                    pbar.set_description(f"Processing {im_path}")

                    im = io.imread(im_path)
                    gt = io.imread(gt_path)
                    if len(gt.shape) > 2:
                        gt = gt.mean(axis=2)
                    scale = 2048 / max(*im.shape[:2])
                    if scale < 1:
                        im = cv2.resize(im, (0,0), fx=scale, fy=scale) 
                        gt = cv2.resize(gt, (0,0), fx=scale, fy=scale)
                    
                    if len(im.shape) < 3:
                        im = im[:, :, np.newaxis]
                    im_shp=im.shape[0:2]
                    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
                    im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
                    image = torch.divide(im_tensor,255.0)
                    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

                    if torch.cuda.is_available():
                        image=image.cuda()
                    result=net(image)
                    result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
                    ma = torch.max(result)
                    mi = torch.min(result)
                    result = (result-mi)/(ma-mi)
                    res = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
                    res = res[:,:,0]

                    if i in to_vis:
                        im_name=im_path.split('/')[-1].split('.')[0]
                        im_path = str(dataset_vis / f'{im_name}.png')
                        res_vis = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
                        gt_vis = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
                        # im_vis = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                        vis = np.hstack((im, gt_vis, res_vis))
                        io.imsave(im_path, vis)
                        # print(im_path)

                    th = np.uint8(127)
                    res = res > th
                    gt = gt > th
                    pre, rec, f1 = get_pr_rc_f1(res, gt)
                    iou = get_IoU(res, gt)
                    print(pre, rec, f1, iou)
                    for pos, val in enumerate([pre, rec, f1, iou]):
                        results[dataset["name"]][pos].append(val)
                results[dataset["name"]] = np.mean(results[dataset["name"]], axis=1)
        df = pd.DataFrame(results)
        df.index = ["Precision", "Recall", "F1 Score", "IoU"]
        df = df.T
        print(df)
        df.to_csv(result_path / "results.csv")

        

