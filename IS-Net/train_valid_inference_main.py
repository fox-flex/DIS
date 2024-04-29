import os
import time
import numpy as np
from skimage import io
import time
from tqdm.auto import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from basics import  f1_mae_torch #normPRED, GOSPRF1ScoresCache,f1score_torch,
from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize #GOSDatasetCache,



def get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar, train_dataloaders_val, train_datasets_val):

    # train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
    #                                                      cache_size = hypar["cache_size"],
    #                                                      cache_boost = hypar["cache_boost_train"],
    #                                                      my_transforms = [
    #                                                                      GOSRandomHFlip(),
    #                                                                      # GOSResize(hypar["input_size"]),
    #                                                                      # GOSRandomCrop(hypar["crop_size"]),
    #                                                                       GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
    #                                                                       ],
    #                                                      batch_size = hypar["batch_size_train"],
    #                                                      shuffle = True)
    device = hypar["device"]
    torch.manual_seed(hypar["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar["seed"])

    print("define gt encoder ...")
    net = ISNetGTEncoder() #UNETGTENCODERCombine()
    ## load the existing model gt encoder
    if(hypar["gt_encoder_model"]!=""):
        model_path = hypar["model_path"]+"/"+hypar["gt_encoder_model"]
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path,map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return net ############

    if torch.cuda.is_available():
        net.cuda()

    print("--- define optimizer for GT Encoder---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model_path = hypar["model_path"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"] # count the total iteration number
    ite_num4val = 0 #
    running_loss = 0.0 # count the toal loss
    running_tar_loss = 0.0 # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num): ## set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if(ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            labels = data['label']

            if(hypar["model_digit"]=="full"):
                labels = labels.type(torch.FloatTensor)
            else:
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                labels_v = Variable(labels, requires_grad=False)

            # print("time lapse for data preparation: ", time.time()-start_read, ' s')

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            ds, fs = net(labels_v)#net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time()-start_inf_loss_back

            print("GT Encoder Training>>>"+model_path.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, time.time()-start_last, time.time()-start_last-end_inf_loss_back))
            start_last = time.time()

        notgood_cnt += 1
        # net.eval()
        # tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(net, valid_dataloaders, valid_datasets, hypar, epoch)
        tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(net, train_dataloaders_val, train_datasets_val, hypar, epoch)

        net.train()  # resume train

        tmp_out = 0
        print("last_f1:",last_f1)
        print("tmp_f1:",tmp_f1)
        for fi in range(len(last_f1)):
            if(tmp_f1[fi]>last_f1[fi]):
                tmp_out = 1
        print("tmp_out:",tmp_out)
        if(tmp_out):
            notgood_cnt = 0
            last_f1 = tmp_f1
            tmp_f1_str = [str(round(f1x,4)) for f1x in tmp_f1]
            tmp_mae_str = [str(round(mx,4)) for mx in tmp_mae]
            maxf1 = '_'.join(tmp_f1_str)
            meanM = '_'.join(tmp_mae_str)
            # .cpu().detach().numpy()
            model_name = "/GTENCODER-gpu_itr_"+str(ite_num)+\
                        "_traLoss_"+str(np.round(running_loss / ite_num4val,4))+\
                        "_traTarLoss_"+str(np.round(running_tar_loss / ite_num4val,4))+\
                        "_valLoss_"+str(np.round(val_loss /(i_val+1),4))+\
                        "_valTarLoss_"+str(np.round(tar_loss /(i_val+1),4)) + \
                        "_maxF1_" + maxf1 + \
                        "_mae_" + meanM + \
                        "_time_" + str(np.round(np.mean(np.array(tmp_time))/batch_size_valid,6))+".pth"
            torch.save(net.state_dict(), model_path + model_name)

        running_loss = 0.0
        running_tar_loss = 0.0
        ite_num4val = 0

        if(tmp_f1[0]>0.99):
            print("GT encoder is well-trained and obtained...")
            return net

        if(notgood_cnt >= hypar["early_stop"]):
            print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
            exit()

    print("Training Reaches The Maximum Epoch Number")
    return net

def valid_gt_encoder(net, valid_dataloaders, valid_datasets, hypar, epoch=0):
    device = hypar["device"]
    net.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0


    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()
    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0,256)
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))

        val_cnt = 0.0
        i_val = None

        for i_val, data_val in enumerate(valid_dataloader):

            # imidx_val, inputs_val, labels_val, shapes_val = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape']
            imidx_val, labels_val, shapes_val = data_val['imidx'], data_val['label'], data_val['shape']

            if(hypar["model_digit"]=="full"):
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_val_v = Variable(labels_val.cuda(), requires_grad=False)
            else:
                labels_val_v = Variable(labels_val,requires_grad=False)

            t_start = time.time()
            ds_val = net(labels_val_v)[0]
            t_end = time.time()-t_start
            tmp_time.append(t_end)

            # loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)
            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar["batch_size_valid"]):
                val_cnt = val_cnt + 1.0
                print("num of val: ", val_cnt)
                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[0][t,:,:,:] # B x 1 x H x W

                ## recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[t][0],shapes_val[t][1]),mode='bilinear'))

                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val-mi)/(ma-mi) # max = 1
                # pred_val = normPRED(pred_val)

                gt = np.squeeze(io.imread(valid_dataset.dataset["ori_gt_path"][i_test])) # max = 255
                if gt.max()==1:
                    gt=gt*255
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre,rec,f1,mae = f1_mae_torch(pred_val*255, gt, valid_dataset, i_test, mybins, hypar)

                PRE[i_test,:]=pre
                REC[i_test,:] = rec
                F1[i_test,:] = f1
                MAE[i_test] = mae

            del ds_val, gt
            gc.collect()
            torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()#data[0]
            tar_loss += loss2_val.item()#data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f"% (i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1[i_test,:]), MAE[i_test],t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)
        # print('--------------:', np.mean(f1_m))
        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        print("The max F1 Score: %f"%(np.max(f1_m)))
        print("MAE: ", np.mean(MAE))

    # print('[epoch: %3d/%3d, ite: %5d] tra_ls: %3f, val_ls: %3f, tar_ls: %3f, maxf1: %3f, val_time: %6f'% (epoch + 1, epoch_num, ite_num, running_loss / ite_num4val, val_loss/val_cnt, tar_loss/val_cnt, tmp_f1[-1], time.time()-start_valid))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time

def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar,train_dataloaders_val, train_datasets_val): 
    device = hypar["device"]
    
    if hypar["interm_sup"]:
        print("Get the gt encoder ...")
        featurenet = get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar,train_dataloaders_val, train_datasets_val)
        ## freeze the weights of gt encoder
        for param in featurenet.parameters():
            param.requires_grad=False


    model_path = hypar["model_path"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"] # count the toal iteration number
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num): ## set the epoch num as 100000
        ite_num4val = 0 #
        running_loss = 0.0 # count the toal loss
        running_tar_loss = 0.0 # count the target output loss
        with tqdm(enumerate(gos_dataloader), total=len(gos_dataloader)) as pbar:

            for i, data in pbar:

                if(ite_num >= max_ite):
                    print("Training Reached the Maximal Iteration Number ", max_ite)
                    exit()

                # start_read = time.time()
                ite_num = ite_num + 1
                ite_num4val = ite_num4val + 1

                # get the inputs
                inputs, labels = data['image'], data['label']

                if(hypar["model_digit"]=="full"):
                    inputs = inputs.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)
                else:
                    inputs = inputs.type(torch.HalfTensor)
                    labels = labels.type(torch.HalfTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
                else:
                    inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

                # print("time lapse for data preparation: ", time.time()-start_read, ' s')

                # y zero the parameter gradients
                start_inf_loss_back = time.time()
                optimizer.zero_grad()

                if hypar["interm_sup"]:
                    # forward + backward + optimize
                    ds,dfs = net(inputs_v)
                    _,fs = featurenet(labels_v) ## extract the gt encodings
                    loss2, loss = net.compute_loss_kl(ds, labels_v, dfs, fs, mode='MSE')
                else:
                    # forward + backward + optimize
                    ds,_ = net(inputs_v)
                    loss2, loss = net.compute_loss(ds, labels_v)

                loss.backward()
                optimizer.step()

                # # print statistics
                running_loss += loss.item()
                running_tar_loss += loss2.item()

                # del outputs, loss
                del ds, loss2, loss
                end_inf_loss_back = time.time()-start_inf_loss_back

                gen_num_zeros = lambda x: int(np.ceil(np.log10(x))) if x != 0 else 1
                    #    f"batch: {(i + 1) * batch_size_train:0{gen_num_zeros(train_num)}d}/{train_num}] " \

                desc = f"[train] {model_path.split('/')[-1]}:[E#{epoch + 1:0{gen_num_zeros(epoch_num)}d}/{epoch_num}, " \
                       f"loss:{running_loss / ite_num4val:.3f}, tar:{running_tar_loss / ite_num4val:.3f}"
                pbar.set_description(desc)

                start_last = time.time()
        
        # validate
        notgood_cnt += 1
        net.eval()
        tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(net, valid_dataloaders, valid_datasets, hypar, epoch)
        net.train()  # resume train

        tmp_out = 0
        print("last_f1:",last_f1)
        print("tmp_f1:",tmp_f1)
        for fi in range(len(last_f1)):
            if(tmp_f1[fi]>last_f1[fi]):
                tmp_out = 1
        print("tmp_out:",tmp_out)
        if(tmp_out):
            notgood_cnt = 0
            last_f1 = tmp_f1
            tmp_f1_str = [str(round(f1x,4)) for f1x in tmp_f1]
            tmp_mae_str = [str(round(mx,4)) for mx in tmp_mae]
            maxf1 = '_'.join(tmp_f1_str)
            meanM = '_'.join(tmp_mae_str)
            # .cpu().detach().numpy()
            model_name = "/gpu_itr_"+str(ite_num)+\
                        "_traLoss_"+str(np.round(running_loss / ite_num4val,4))+\
                        "_traTarLoss_"+str(np.round(running_tar_loss / ite_num4val,4))+\
                        "_valLoss_"+str(np.round(val_loss /(i_val+1),4))+\
                        "_valTarLoss_"+str(np.round(tar_loss /(i_val+1),4)) + \
                        "_maxF1_" + maxf1 + \
                        "_mae_" + meanM + \
                        "_time_" + str(np.round(np.mean(np.array(tmp_time))/batch_size_valid,6))+".pth"
            torch.save(net.state_dict(), model_path + model_name)

        # early stopping
        if(notgood_cnt >= hypar["early_stop"]):
            print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
            exit()

    print("Training Reaches The Maximum Epoch Number")

def valid(net, valid_dataloaders, valid_datasets, hypar, epoch=0, vis=False):
    device = hypar["device"]
    net.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0,256)
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))

        with tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)) as pbar:
            for i_val, data_val in pbar:
                val_cnt = val_cnt + 1.0
                imidx_val, inputs_val, labels_val, shapes_val = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape']

                if(hypar["model_digit"]=="full"):
                    inputs_val = inputs_val.type(torch.FloatTensor)
                    labels_val = labels_val.type(torch.FloatTensor)
                else:
                    inputs_val = inputs_val.type(torch.HalfTensor)
                    labels_val = labels_val.type(torch.HalfTensor)

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(labels_val.cuda(), requires_grad=False)
                else:
                    inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,requires_grad=False)

                t_start = time.time()
                ds_val = net(inputs_val_v)[0]
                t_end = time.time()-t_start
                tmp_time.append(t_end)

                # loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)
                loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

                # compute F measure
                for t in range(hypar["batch_size_valid"]):
                    i_test = imidx_val[t].data.numpy()

                    pred_val = ds_val[0][t,:,:,:] # B x 1 x H x W

                    ## recover the prediction spatial size to the orignal image size
                    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[t][0],shapes_val[t][1]),mode='bilinear'))

                    # pred_val = normPRED(pred_val)
                    ma = torch.max(pred_val)
                    mi = torch.min(pred_val)
                    pred_val = (pred_val-mi)/(ma-mi) # max = 1

                    if len(valid_dataset.dataset["ori_gt_path"]) != 0:
                        gt = np.squeeze(io.imread(valid_dataset.dataset["ori_gt_path"][i_test])) # max = 255
                        if gt.max()==1:
                            gt=gt*255
                    else:
                        gt = np.zeros((shapes_val[t][0],shapes_val[t][1]))
                    with torch.no_grad():
                        gt = torch.tensor(gt).to(device)

                    pre,rec,f1,mae = f1_mae_torch(pred_val*255, gt, valid_dataset, i_test, mybins, hypar)


                    PRE[i_test,:]=pre
                    REC[i_test,:] = rec
                    F1[i_test,:] = f1
                    MAE[i_test] = mae

                    del ds_val, gt
                    gc.collect()
                    torch.cuda.empty_cache()

                # if(loss_val.data[0]>1):
                val_loss += loss_val.item()#data[0]
                tar_loss += loss2_val.item()#data[0]

                desc = f"[validating] val_ls:{val_loss / (i_val + 1):.3f}, tar_ls: {tar_loss / (i_val + 1):.3f}, " \
                       f"f1: {np.amax(F1[i_test,:]):.5f}, mae: {MAE[i_test]:.5f}"
                pbar.set_description(desc)
                del loss2_val, loss_val

        # print('============================')
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time

def main(train_datasets,
         valid_datasets,
         hypar): # model: "train", "test"
    device = hypar["device"]

    ### --- Step 1: Build datasets and dataloaders ---
    dataloaders_train = []
    dataloaders_valid = []

    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        ## collect training dataset
        train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        ## build dataloader for training datasets
        train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                             GOSRandomHFlip(), ## this line can be uncommented for horizontal flip augmetation
                                                                             # GOSResize(hypar["input_size"]),
                                                                             # GOSRandomCrop(hypar["crop_size"]), ## this line can be uncommented for randomcrop augmentation
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_train"],
                                                             shuffle = True,
                                                             flag="train")
        train_dataloaders_val, train_datasets_val = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_valid"],
                                                             shuffle = False,
                                                             flag="train")
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    ## build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    ## build dataloader for training datasets
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                          cache_size = hypar["cache_size"],
                                                          cache_boost = hypar["cache_boost_valid"],
                                                          my_transforms = [
                                                                           GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                           # GOSResize(hypar["input_size"])
                                                                           ],
                                                          batch_size=hypar["batch_size_valid"],
                                                          shuffle=False,
                                                          flag="valid")
    print(len(valid_dataloaders), " valid dataloaders created")
    # print(valid_datasets[0]["data_name"])

    ### --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net = hypar["model"]#GOSNETINC(3,1)
    net.load_state_dict(torch.load("../ckpt/isnet-general-use.pth"))

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if(hypar["restore_model"]!=""):
        print("restore model from:")
        model_path = hypar["restore_model"]
        print(model_path)
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
        else:
            net.load_state_dict(torch.load(model_path,map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    ### --- Step 3: Train or Valid Model ---
    if(hypar["mode"]=="train"):
        train(net,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              hypar,
              train_dataloaders_val, train_datasets_val)
    else:
        return valid(net,
              valid_dataloaders,
              valid_datasets,
              hypar)
