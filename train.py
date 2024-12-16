# Copyright 2020 by Jingchao Hou.  

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import torch
import imageio
import numpy as np
import math
import sys
import os
import torch.nn as nn
import torch.nn.functional as F

from load_data import LoadData, LoadVisualData
from msssim import MSSSIM
from model import PyNET
from vgg import vgg_19
from utils import normalize_batch, process_command_args

 

import argparse
import warnings
warnings.filterwarnings("ignore")



to_image = transforms.Compose([transforms.ToPILImage()])

np.random.seed(0)
torch.manual_seed(0)


# Processing command arguments
level, batch_size, learning_rate, restore_epoch, num_train_epochs, dataset_dir = process_command_args(sys.argv)

sys.argv=[sys.argv[0]]+sys.argv[5:]


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--choose", type =str,default='all', help="increase output verbosity")
parser.add_argument("--modelchoose", type =str,default='all', help="choose the model")
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--scales', type=int, default=3, help='Scales')
parser.add_argument('--level', type=int, default=3, help='Level')
parser.add_argument('--inter_nums', type=int, default=4, help='Intermediate numbers')
args = parser.parse_args()
print(args.choose)

dslr_scale = float(1) / (2 ** (level - 1))
# Dataset size

TRAIN_SIZE = 46839
TEST_SIZE = 1204

if 'ISPW' in dataset_dir:
    TRAIN_SIZE=27300 # 28800-1500 = 27300
    TEST_SIZE=1500
elif 'xiaomi' in dataset_dir:
    TRAIN_SIZE=9792 # 28800-1500 = 27300
    TEST_SIZE=1224


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def create_window(self, window_size, channel=1):
        def gaussian(window_size, sigma):
            window_range = torch.arange(window_size, dtype=torch.float) - window_size // 2
            gauss = torch.exp(-window_range.pow(2) / (2 * sigma ** 2))
            return gauss / gauss.sum()


        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, y_pred, y_true):
        (_, channel, _, _) = y_pred.size()
        if channel == self.channel and self.window.data.type() == y_pred.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel=y_pred.size(1))
            window = window.to(y_pred.device)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(y_pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def val_psnr_ssim(model,test_loader,device):
    model.eval()
    MS_SSIM = MSSSIM()
    ssim_r = SSIMLoss()
    MSE_loss = torch.nn.MSELoss()
    loss_mse_eval=0
    loss_psnr_eval=0
    loss_ssim=0
    #print('test_loader_len',len(test_loader))
    #os.exit()
    with torch.no_grad():
        for j,raw_image in enumerate(test_loader):
            torch.cuda.empty_cache()
            raw_image,y, =raw_image
            raw_image = raw_image.to(device)
            y=y.to(device)
            raw_image = raw_image.to(device)
            enhanced = model(raw_image.detach())
            #print(enhanced.shape,y.shape)
            loss_mse_temp = MSE_loss(enhanced, y).item()
            loss_mse_eval += loss_mse_temp
            loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
            #print(enhanced.shape,y.shape)
            loss_ssim += MS_SSIM(enhanced, y)
        loss_mse_eval = loss_mse_eval / TEST_SIZE
        loss_psnr_eval = loss_psnr_eval / TEST_SIZE
        loss_ssim = loss_ssim / TEST_SIZE
    #print('val_metrics:',loss_mse_eval,loss_psnr_eval,loss_ssim)
    return loss_mse_eval,loss_psnr_eval,loss_ssim
def test_model(generator,test_loader,device,epoch):
    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()
 
    ssim_r = SSIMLoss()
    # Evaluate the model
    loss_mse_eval = 0
    loss_psnr_eval = 0
    loss_vgg_eval = 0
    loss_ssim_eval = 0
    loss_ssim =0
    generator.eval()
    with torch.no_grad():
        test_iter = iter(test_loader)
        for j in range(len(test_loader)):
            x, y = next(test_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            #print(x.shape)
            enhanced = generator(x)
            #print(x.shape,enhanced.shape,y.shape)
            loss_mse_temp = MSE_loss(enhanced, y).item()
            loss_mse_eval += loss_mse_temp
            #print(loss_mse_temp,20 * math.log10(1.0 / math.sqrt(loss_mse_temp)))
            loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
            #print(loss_mse_temp,20 * math.log10(1.0 / math.sqrt(loss_mse_temp)))
            loss_ssim_eval += MS_SSIM(y, enhanced)
            loss_ssim += ssim_r(enhanced, y)
            enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
            target_vgg_eval = VGG_19(normalize_batch(y)).detach()
            loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()
    #print(TEST_SIZE,len(test_iter))
    loss_mse_eval = loss_mse_eval / TEST_SIZE
    loss_psnr_eval = loss_psnr_eval / TEST_SIZE
    loss_vgg_eval = loss_vgg_eval / TEST_SIZE
    loss_ssim_eval = loss_ssim_eval / TEST_SIZE
    loss_ssim =loss_ssim / TEST_SIZE
    print("Epoch %d, mse_test: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f, ssim: %.4f" % (epoch,
                loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval, loss_ssim))

def train_model():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,5,6,7"
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Creating dataset loaders

    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, dslr_scale, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    visual_dataset = LoadVisualData(dataset_dir, 10, dslr_scale, level)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating image processing network and optimizer

    generator = PyNET(level=level, instance_norm=True, instance_norm_level_1=True).to(device)
    generator = torch.nn.DataParallel(generator)

    optimizer = Adam(params=generator.parameters(), lr=learning_rate)

    # Restoring the variables

    if level < 5:
        if level ==0:
            generator.load_state_dict(torch.load("models/pynet_level_0_epoch_49.pth"), strict=True)
        else:
            generator.load_state_dict(torch.load("models/pynet_level_" + str(level + 1) +
                                            "_epoch_" + str(restore_epoch) + ".pth"), strict=False)

    # Losses

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()
    

    # Train the network

    for epoch in range(num_train_epochs):
        torch.cuda.empty_cache()

        train_iter = iter(train_loader)
        for i in range(len(train_loader)):

            optimizer.zero_grad()
            x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            enhanced = generator(x)

            # MSE Loss
            loss_mse = MSE_loss(enhanced, y)

            # VGG Loss

            if level < 5:
                enhanced_vgg = VGG_19(normalize_batch(enhanced))
                target_vgg = VGG_19(normalize_batch(y))
                loss_content = MSE_loss(enhanced_vgg, target_vgg)

            # Total Loss

            if level == 5 or level == 4:
                total_loss = loss_mse
            if level == 3 or level == 2:
                total_loss = loss_mse * 10 + loss_content
            if level == 1:
                total_loss = loss_mse * 10 + loss_content
            if level == 0:
                loss_ssim = MS_SSIM(enhanced, y)
                total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

            # Perform the optimization step

            total_loss.backward()
            optimizer.step()

            if i == 0:

                # Save the model that corresponds to the current epoch

                generator.eval().cpu()
                torch.save(generator.state_dict(), "models/pynet_level_" + str(level) + "_epo_" + str(epoch) + ".pth")
                generator.to(device).train()

                # Save visual results for several test images

                generator.eval()
                with torch.no_grad():

                    visual_iter = iter(visual_loader)
                    for j in range(len(visual_loader)):

                        torch.cuda.empty_cache()

                        raw_image = next(visual_iter)
                        raw_image = raw_image.to(device, non_blocking=True)

                        enhanced = generator(raw_image.detach())
                        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                        imageio.imwrite("results/pynet_img_" + str(j) + "_level_" + str(level) + "_epoch_" +
                                        str(epoch) + ".jpg", enhanced)

                # Evaluate the model

                loss_mse_eval = 0
                loss_psnr_eval = 0
                loss_vgg_eval = 0
                loss_ssim_eval = 0

                generator.eval()
                with torch.no_grad():

                    test_iter = iter(test_loader)
                    for j in range(len(test_loader)):

                        x, y = next(test_iter)
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        enhanced = generator(x)

                        loss_mse_temp = MSE_loss(enhanced, y).item()

                        loss_mse_eval += loss_mse_temp
                        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

                        if level < 2:
                            loss_ssim_eval += MS_SSIM(y, enhanced)

                        if level < 5:
                            enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                            target_vgg_eval = VGG_19(normalize_batch(y)).detach()

                            loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()

                loss_mse_eval = loss_mse_eval / TEST_SIZE
                loss_psnr_eval = loss_psnr_eval / TEST_SIZE
                loss_vgg_eval = loss_vgg_eval / TEST_SIZE
                loss_ssim_eval = loss_ssim_eval / TEST_SIZE

                if level < 2:
                    print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f" % (epoch,
                            loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval))
                elif level < 5:
                    print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f" % (epoch,
                            loss_mse_eval, loss_psnr_eval, loss_vgg_eval))
                else:
                    print("Epoch %d, mse: %.4f, psnr: %.4f" % (epoch, loss_mse_eval, loss_psnr_eval))

                generator.train()


def transfer_weights(model_old, model_new):
    # 获取两个模型的权重字典
    old_weights = model_old.state_dict()
    new_weights = model_new.state_dict()

    # 遍历旧模型的权重字典，如果键在新模型的权重字典中也存在，则复制权重
    for key in old_weights:
        if key in new_weights and old_weights[key].shape == new_weights[key].shape:
            new_weights[key] = old_weights[key]

    # 将更新后的权重字典加载到新模型中
    return model_new.load_state_dict(new_weights)

def train_my(train_loader,test_loader,visual_loader,generator,model_name,restore_epoch=0):
    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()
    
    ssim_r = SSIMLoss()
    # Train the network
    test_model(generator,test_loader,device,epoch=0)

    for epoch in range(num_train_epochs):
        epoch=epoch+restore_epoch
        torch.cuda.empty_cache()
        generator.to(device).train()
        mse_loss_train=0
        train_iter = iter(train_loader)
        for i in range(len(train_loader)):
            optimizer.zero_grad()
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            enhanced = generator(x)
            # MSE Loss
            loss_mse = MSE_loss(enhanced, y)
            # VGG Loss
            enhanced_vgg = VGG_19(normalize_batch(enhanced))
            target_vgg = VGG_19(normalize_batch(y))
            loss_content = MSE_loss(enhanced_vgg, target_vgg)
            # Total Loss
            loss_ssim = MS_SSIM(enhanced, y)
            #ssim_r = ssim_r(enhanced, y)
            total_loss = loss_mse
            #total_loss = loss_mse + loss_content
            #total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4
            # Perform the optimization step
            total_loss.backward()

            optimizer.step()
            #scheduler.step()
            mse_loss_train =mse_loss_train+loss_mse.item()
        # Save the model that corresponds to the current epoch
        generator.eval().cpu()
        if args.choose=='all':
            if not os.path.exists("models/"+model_name):
                os.makedirs("models/"+model_name)
            if 'xiaomi' in dataset_dir:
                torch.save(generator.state_dict(), "models/"+model_name+"/ISP_xiaomi" + "_epoch_" + str(epoch) + ".pth")
            elif 'ISPW' in dataset_dir:
                torch.save(generator.state_dict(), "models/"+model_name+"/ISP_select" + "_epoch_" + str(epoch) + ".pth")
            else:
                torch.save(generator.state_dict(), "models/"+model_name+"/ISP_zrr" + "_epoch_" + str(epoch) + ".pth")
        else:
            pth_path = 'models/ablation/'+args.choose
            if not os.path.exists(pth_path):
                os.makedirs(pth_path)
            torch.save(generator.state_dict(), pth_path+"/ISP_select" + "_epoch_" + str(epoch) + ".pth")
        generator.to(device).train()
        # Save visual results for several test images
        generator.eval()
        #with torch.no_grad():
        #    visual_iter = iter(visual_loader)
        #    for j in range(len(visual_loader)):
        #        torch.cuda.empty_cache()
        #        raw_image = next(visual_iter)
        #        raw_image = raw_image.to(device, non_blocking=True)
        #        enhanced = generator(raw_image.detach())
        #        enhanced =torch.clamp(enhanced,0,1)
        #        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))
        #        imageio.imwrite("results/ISP_select_img_" + str(j)  + "_epoch_" +
        #                        str(epoch) + ".jpg", enhanced)
        # Evaluate the model
        loss_mse_eval = 0
        loss_psnr_eval = 0
        loss_vgg_eval = 0
        loss_ssim_eval = 0
        loss_ssim =0
        generator.eval()
        with torch.no_grad():
            test_iter = iter(test_loader)
            for j in range(len(test_loader)):
                x, y = next(test_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                enhanced = generator(x)
                loss_mse_temp = MSE_loss(enhanced, y).item()
                loss_mse_eval += loss_mse_temp
                loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
                loss_ssim_eval += MS_SSIM(y, enhanced)
                #print(loss_mse_temp,20 * math.log10(1.0 / math.sqrt(loss_mse_temp)))
                loss_ssim += ssim_r(enhanced, y)
                enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                target_vgg_eval = VGG_19(normalize_batch(y)).detach()
                loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()
        loss_mse_eval = loss_mse_eval / TEST_SIZE
        loss_psnr_eval = loss_psnr_eval / TEST_SIZE
        loss_vgg_eval = loss_vgg_eval / TEST_SIZE
        loss_ssim_eval = loss_ssim_eval / TEST_SIZE
        loss_ssim =loss_ssim / TEST_SIZE
        print("Epoch %d, mse_train: %.4f, mse_test: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f, ssim: %.4f" % (epoch, mse_loss_train/len(train_loader),
                    loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval, loss_ssim))
        generator.train()

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from each key in the state_dict.
    """
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return new_state_dict

if __name__ == '__main__':
    # 使用 parser 获取其他参数
    import model_raw2rgb
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,4" 
    torch.backends.cudnn.deterministic = True
    # 如果想确保CuDNN的行为是确定的，可以添加：
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
    # Creating image processing network and optimizer
    layer_4=True
    restore_epoch=0
    if args.modelchoose =='proposed':
        generator = model_raw2rgb.ISP_select(num_layers=7, window_size=7, rggb_kernel_size=3, affine_init="xavier", attention_kernel_size=5,scales=7,combine_method='sum',ablation = args.choose).to(device)
        #generator =model_raw2rgb.MultiScaleISP().to(device)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=num_train_epochs)
        #generator.load_state_dict(torch.load("models/ISP_select_newall_2" + ".pth"), strict=True)
        model_path = ["models/ISP_select_epoch_49.pth","models/ISP_select_epoch_49.pth","models/ISP_xiaomi_epoch_49.pth"]
        if 'ISPW' in dataset_dir:
            generator.load_state_dict(remove_module_prefix(torch.load(model_path[1])),strict=True)
        elif 'xiaomi' in dataset_dir:
             generator.load_state_dict(remove_module_prefix(torch.load(model_path[2])),strict=True)
        else:
             generator.load_state_dict(remove_module_prefix(torch.load(model_path[0])),strict=True)
        generator.load_state_dict(remove_module_prefix(torch.load("models/ISP_select_epoch_49" + ".pth")), strict=True)
 
        
    generator = torch.nn.DataParallel(generator)
    generator.to(device)
    optimizer = Adam(params=generator.parameters(), lr=learning_rate)
    # Creating dataset loaders
    
    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, dslr_scale, test=False, layer_4=layer_4)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=16,
                              pin_memory=True, drop_last=True)
    test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True,layer_4=layer_4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)
    visual_dataset = LoadVisualData(dataset_dir, 10, dslr_scale, level,layer_4=layer_4)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)
    print(TEST_SIZE)
    val_psnr_ssim(generator,test_loader,device)
    if 'xiaomi' in dataset_dir or 'ISPW' in dataset_dir:
        train_my(train_loader,test_loader,visual_loader,generator,args.modelchoose,restore_epoch=restore_epoch)
    elif "zurich" in dataset_dir:
        train_my(train_loader,test_loader,visual_loader,generator,args.modelchoose)
    else:
        train_model()

