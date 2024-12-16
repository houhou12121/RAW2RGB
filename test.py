# Copyright 2024 by Jingchao hou.  

from scipy import misc
import numpy as np
import sys
import os

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from load_data import LoadVisualData, LoadData
 
 
import utils
import imageio
from torchsummary import summary
from thop import profile

import math
from train_all import SSIMLoss as SSIMLoss
import argparse
from msssim import MSSSIM

print('start')
np.random.seed(0)
torch.manual_seed(0)

global dataset_dir
global TEST_SIZE

to_image = transforms.Compose([transforms.ToPILImage()])

level, restore_epoch, dataset_dir, use_gpu, orig_model = utils.process_test_model_args(sys.argv)
use_gpu = False
dslr_scale = float(1) / (2 ** (level - 1))
dslr_scale =2.0



parser = argparse.ArgumentParser()
parser.add_argument("-c", "--choose", type =str,default='all', help="increase output verbosity")
parser.add_argument("--modelchoose", type =str,default='all', help="choose the model")
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--scales', type=int, default=3, help='Scales')
parser.add_argument('--level', type=int, default=3, help='Level')
parser.add_argument('--inter_nums', type=int, default=4, help='Intermediate numbers')
args = parser.parse_args()




MSE_loss = torch.nn.MSELoss()
def psnr(loss_mse):
    loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse))
    return loss_psnr_eval




def val_psnr_ssim(model,test_loader):
    model.eval()
    MS_SSIM = MSSSIM()
    ssim_r = SSIMLoss()
    loss_mse_eval=0
    loss_psnr_eval=0
    loss_ssimm=0
    loss_ssim=0
    with torch.no_grad():
        for j,raw_image in enumerate(test_loader):
            torch.cuda.empty_cache()
            raw_image,y, =raw_image
            raw_image = raw_image.to(device)
            y=y.to(device)
            model.to(device)
            #print(raw_image.shape,y.shape)
            if use_gpu == "true":
                raw_image = raw_image.to(device, dtype=torch.half)
            else:
                raw_image = raw_image.to(device)
            if data_name == 'ZRR_full':
                # 获取 raw_image 的高度和宽度
                _, _, h, w = raw_image.shape
                # 计算需要填充的大小
                pad_h = (64 - h % 64) % 64
                pad_w = (64 - w % 64) % 64
                # 如果需要填充，则进行填充
                if pad_h > 0 or pad_w > 0:
                    raw_image = torch.nn.functional.pad(raw_image, (0, pad_w, 0, pad_h), mode='reflect')
                # Run inference
                 # 打印 raw_image 和 model 的设备信息
                #print(f"Raw image is on device: {raw_image.device}")
                #print(f"Model is on device: {next(model.parameters()).device}")
                enhanced = model(raw_image.detach())
                # 如果进行了填充，则将 enhanced 裁剪回原始大小
                if pad_h > 0 or pad_w > 0:
                    enhanced = enhanced[:, :, :2*h, :2*w]
            else:
                # Run inference
                enhanced = model(raw_image.detach())
            #print(enhanced.shape,y.shape)
            loss_mse_temp = MSE_loss(enhanced, y).item()
            loss_mse_eval += loss_mse_temp
            loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))
            loss_ssimm += MS_SSIM(enhanced, y)
            loss_ssim += ssim_r(enhanced,y)
        loss_mse_eval = loss_mse_eval / TEST_SIZE
        loss_psnr_eval = loss_psnr_eval / TEST_SIZE
        loss_ssimm = loss_ssimm / TEST_SIZE
        loss_ssim = loss_ssim / TEST_SIZE
    return loss_mse_eval,loss_psnr_eval,loss_ssimm,loss_ssim



def test_model(model,data_name,write_png=True,layer_4=True,dslr_scale = 2,model_name=None):
    import model_raw2rgb
    #print("CUDA visible devices: " + str(torch.cuda.device_count()))
    #print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
    # Creating dataset loaders
    # Dataset size
    #TEST_SIZE = 900
    test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True,layer_4=layer_4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # Creating and loading pre-trained PyNET model
    if use_gpu == "true":
        model.half()
    model.eval()
    model.to(device)
    # Processing full-resolution RAW images
    mse,psnr,ssimm,ssim = val_psnr_ssim(model,test_loader)
    print('mse,psnr,ssim: ',mse,psnr,ssim)
    if write_png:
        with torch.no_grad():
            #visual_iter = iter(test_loader)
            print(len(test_loader))
            print('start')
            for j,raw_image in enumerate(test_loader):
                print('start2')
                print("Processing image " + str(j))
                torch.cuda.empty_cache()
                #raw_image = next(visual_iter)
                raw_image,label, =raw_image
    
                if use_gpu == "true":
                    raw_image = raw_image.to(device, dtype=torch.half)
                else:
                    raw_image = raw_image.to(device)
    
                # Run inference
                #print(raw_image)
                if data_name == 'ZRR_full':
                    # 获取 raw_image 的高度和宽度
                    _, _, h, w = raw_image.shape
                    # 计算需要填充的大小
                    pad_h = (64 - h % 64) % 64
                    pad_w = (64 - w % 64) % 64
                    # 如果需要填充，则进行填充
                    if pad_h > 0 or pad_w > 0:
                        raw_image = torch.nn.functional.pad(raw_image, (0, pad_w, 0, pad_h), mode='reflect')
                    # Run inference
                    enhanced = model(raw_image.detach())
                    # 如果进行了填充，则将 enhanced 裁剪回原始大小
                    if pad_h > 0 or pad_w > 0:
                        enhanced = enhanced[:, :, :2*h, :2*w]
                else:
                    # Run inference
                    enhanced = model(raw_image.detach())

                enhanced = model(raw_image.detach())
                enhanced =torch.clamp(enhanced,0,1)
                #print(enhanced)
                enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))
                #print(enhanced)
    
                # Save the results as .png images
                if not os.path.exists("./results/"+data_name):
                    os.makedirs("./results/"+data_name)
                print("./results/"+data_name)
                if model_name:
                    if not os.path.exists("./results/"+model_name+"/"+data_name):
                        os.makedirs("./results/"+model_name+"/"+data_name)
                    print("./results/"+model_name+"/"+data_name)
                    imageio.imwrite("results/"+model_name+"/"+data_name+'/' + str(j) + ".png", enhanced)
                else:
                    imageio.imwrite("results/"+data_name+'/' + str(j) + ".png", enhanced)
                #if j>=20:
                #    break
                #misc.imsave("results/zrr/" +str(j) + ".png", enhanced)

def remove_module_prefix(state_dict):
    """
    Removes the 'module.' prefix from each key in the state_dict.
    """
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return new_state_dict


if __name__ == '__main__':
    #global dataset_dir
    #global TEST_SIZE
    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import model_raw2rgb
    test_choose ='UHDformer'
    test_choose == 'LUT_denoiser'
    test_choose = 'LUT'
    test_choose = 'tiny_ISP'
    test_choose ='my'
    print('start')
    data_name =['ZRR','ISPW','phone']
    dataset_dir_all = ['../../dataset/zurich_RAW_RGB','../../dataset/ISPW_RAW_RGB','../../dataset/xiaomi13ultra_RAW_RGB']
    if test_choose == 'my':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        generator = model_raw2rgb.ISP_select(num_layers=7, window_size=7, rggb_kernel_size=3, affine_init="xavier", attention_kernel_size=5,scales=7,combine_method='sum',save_images =True,ablation = 'notattention').to(device)
        generator = torch.nn.DataParallel(generator)
        generator.load_state_dict(torch.load("models/ablation/notattention/ISP_select_epoch_99" + ".pth"), strict=True)
        generator.eval()
        # Creating dataset loaders
        dslr_scale = 2
        TEST_SIZE = 1500
        dataset_dir ='../../dataset/ISPW_RAW_RGB'
        test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                 pin_memory=True, drop_last=False)
        with torch.no_grad():
            #visual_iter = iter(test_loader)
            print(len(test_loader))
            print('start')
            for j,raw_image in enumerate(test_loader):
                print('start2')
                print("Processing image " + str(j))
                torch.cuda.empty_cache()
                #raw_image = next(visual_iter)
                raw_image,label, =raw_image
                if use_gpu == "true":
                    raw_image = raw_image.to(device, dtype=torch.half)
                else:
                    raw_image = raw_image.to(device)
                # Run inference
                #print(raw_image)
                enhanced = generator(raw_image.detach())
                enhanced =torch.clamp(enhanced,0,1)
                #print(enhanced)
                enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))
    elif test_choose =='my_ZRR_full':
        model_path = "models/ISP_select_newall_2.pth"
        data_name = 'ZRR_full' 
        dataset_dir = '../../dataset/zurich_RAW_RGB/test/' 
        TEST_SIZE = 10
        model = model_raw2rgb.ISP_select(num_layers=7, window_size=7, rggb_kernel_size=3, affine_init="xavier", attention_kernel_size=5,scales=7,combine_method='sum',save_images=False).to(device)
        if device !=torch.device("cpu"):
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path), strict=True)
        else:
            state_dict =torch.load(model_path)
            state_dict =remove_module_prefix(state_dict)
            model.load_state_dict(state_dict, strict=True)
        
        test_model(model, data_name)
