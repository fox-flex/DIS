
import os
import warnings
warnings.filterwarnings('ignore')

import torch

from models import *
from train_valid_inference_main import main
from config import (
    hypar, hypar_set_mode,
    dataset_uhrsd_tr, dataset_uhrsd_te,
    dataset_synt_tr, dataset_synt_te,
    dataset_duts_tr, dataset_duts_te
)



if __name__ == "__main__":
    hypar["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## configure the train, valid and inference datasets
    train_datasets = [
        dataset_uhrsd_tr,
        dataset_synt_tr,
        # dataset_duts_tr,
    ]
    valid_datasets = [
        dataset_uhrsd_te,
        dataset_synt_te,
        # dataset_duts_te,
    ]

    hypar["model"] = ISNetDIS() #U2NETFASTFEATURESUP()
    hypar = hypar_set_mode(hypar, 'train')
    hypar["model_path"] ="../saved_models/IS-Net-train-comb"
    hypar["restore_model"] = "../saved_models/isnet-general-use.pth"

    main(
        train_datasets,
        valid_datasets,
        hypar=hypar
    )