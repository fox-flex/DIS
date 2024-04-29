dataset_uhrsd_tr = {
    "name": "UHRSD_TR",
    "im_dir": "../data_raw/UHRSD_TR/image",
    "gt_dir": "../data_raw/UHRSD_TR/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir":"../cache/UHRSD_TR"
}
dataset_uhrsd_te = {
    "name": "UHRSD_TE",
    "im_dir": "../data_raw/UHRSD_TE/image",
    "gt_dir": "../data_raw/UHRSD_TE/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir":"../cache/UHRSD_TE"
}

dataset_duts_tr = {
    "name": "DUTS_TR",
    "im_dir": "../data_raw/DUTS-TR/DUTS-TR-Image",
    "gt_dir": "../data_raw/DUTS-TR/DUTS-TR-Mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir":"../cache/DUTS-TR"
}
dataset_duts_te = {
    "name": "DUTS_TE",
    "im_dir": "../data_raw/DUTS-TE/DUTS-TE-Image",
    "gt_dir": "../data_raw/DUTS-TE/DUTS-TE-Mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir":"../cache/DUTS-TE"
}

dataset_synt_tr = {
    "name": "synthetic_tr",
    "im_dir": "../data_gen/train/image",
    "gt_dir": "../data_gen/train/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir" : "../cache/synthetic_tr"
}
dataset_synt_te = {
    "name": "synthetic_te",
    "im_dir": "../data_gen/valid/image",
    "gt_dir": "../data_gen/valid/mask",
    "im_ext": ".jpg",
    "gt_ext": ".png",
    "cache_dir" : "../cache/synthetic_te"
}

### --------------- STEP 2: Configuring the hyperparamters for Training, validation and inferencing ---------------
hypar = {}

## -- 2.1. configure the model saving or restoring path --
hypar["mode"] = "train"
# hypar["mode"] = "valid"    
## "train": for training,
## "valid": for validation and inferening,
## in "valid" mode, it will calculate the accuracy as well as save the prediciton results into the "hypar["valid_out_dir"]", which shouldn't be ""
## otherwise only accuracy will be calculated and no predictions will be saved
hypar["interm_sup"] = False ## in-dicate if activate intermediate feature supervision

# if hypar["restore_model"]!="":
#     hypar["start_ite"] = int(hypar["restore_model"].split("_")[2])

## -- 2.2. choose floating point accuracy --
hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

## -- 2.3. cache data spatial size --
## To handle large size input images, which take a lot of time for loading in training,
#  we introduce the cache mechanism for pre-convering and resizing the jpg and png images into .pt file
hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size
hypar["cache_boost_train"] = False ## "True" or "False", indicates wheather to load all the training datasets into RAM, True will greatly speed the training process while requires more RAM
hypar["cache_boost_valid"] = False ## "True" or "False", indicates wheather to load all the validation datasets into RAM, True will greatly speed the training process while requires more RAM

## --- 2.4. data augmentation parameters ---
hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
hypar["random_flip_h"] = 1 ## horizontal flip, currently hard coded in the dataloader and it is not in use
hypar["random_flip_v"] = 0 ## vertical flip , currently not in use

## --- 2.5. define model  ---
print("building model...")
hypar["early_stop"] = 20 ## stop the training when no improvement in the past 20 validation periods, smaller numbers can be used here e.g., 5 or 10.

hypar["batch_size_train"] = 8 ## batch size for training
hypar["batch_size_valid"] = 1 ## batch size for validation and inferencing
print("batch size: ", hypar["batch_size_train"])

hypar["max_ite"] = 1000000 ## if early stop couldn't stop the training process, stop it by the max_ite_num
hypar["max_epoch_num"] = 50 ## if early stop and max_ite couldn't stop the training process, stop it by the max_epoch_num
hypar["restore_model"] = "" ## name of the segmentation model weights .pth for resume training process from last stop or for the inferencing

def hypar_set_mode(hypar_, mode_='train'):
    hypar_["mode"] = mode_
    if hypar_["mode"] == "train":
        hypar_["valid_out_dir"] = "" ## for "train" model leave it as "", for "valid"("inference") mode: set it according to your local directory
        hypar_["model_path"] ="../saved_models/IS-Net-train" ## model weights saving (or restoring) path
        hypar_["start_ite"] = 0 ## start iteration for the training, can be changed to match the restored training process
        hypar_["gt_encoder_model"] = ""
    else: ## configure the segmentation output path and the to-be-used model weights path
        hypar_["valid_out_dir"] = "../your-results/"##"../DIS5K-Results-test" ## output inferenced segmentation maps into this fold
        hypar_["model_path"] = "../saved_models/IS-Net-eval" ## load trained weights from this path
    return hypar_