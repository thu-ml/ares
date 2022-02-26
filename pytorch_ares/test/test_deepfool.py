import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import timm
from torchvision import transforms
os.environ['TORCH_HOME']=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'model')
from torchvision.utils import save_image
import torch
from pytorch_ares.attack_torch import DeepFool
from third_party.example.cifar10.pytorch_cifar10.models import *
from pytorch_ares.dataset_torch.datasets_test import datasets


class imagenet_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.input_size = self.model.default_cfg['input_size'][1]
        self.interpolation = self.model.default_cfg['interpolation']
        self.crop_pct = self.model.default_cfg['crop_pct']
        self.mean=self.model.default_cfg['mean']
        self.std=self.model.default_cfg['std']
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean = torch.tensor(self.mean).view(3,1,1).to(self.device)
        self.std = torch.tensor(self.std).view(3,1,1).to(self.device)   
        x = (x - self.mean) / self.std
        labels = self.model(x.to(self.device))
        return labels

class cifar10_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)
        self.std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)
        x = (x - self.mean_torch_c) / self.std_torch_c
        labels = self.model(x.to(self.device))
        return labels



def test(args):
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    if args.dataset_name == "imagenet":
        model = timm.create_model('resnet50', pretrained=True)
        net = imagenet_model(device,model)
        test_loader = datasets(args.dataset_name, args.batchsize,net.input_size,net.crop_pct,net.interpolation, args.cifar10_path, args.imagenet_val_path, args.imagenet_targrt_path,args.imagenet_path)

        
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'model/checkpoint/resnet18_ckpt.pth')
        model = ResNet18()
        pretrain_dict = torch.load(path, map_location=device)
        model.load_state_dict(pretrain_dict['net'])
        net = cifar10_model(device, model)
        test_loader = datasets(args.dataset_name, args.batchsize, args.input_size, args.crop_pct, args.interpolation, args.cifar10_path, args.imagenet_val_path, args.imagenet_targrt_path,args.imagenet_path)
    net.eval()
    success_num = 0
    test_num= 0
   
    attack = DeepFool(net, args.overshoot, args.max_iter, args.norm, args.target, device)
    name = attack.__class__.__name__
    
    if args.dataset_name == "imagenet":
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels,target_labels)
            
            
            out = net(image)
            out = torch.argmax(out, dim=1)
            test_num += (out == labels).sum()
            if i ==1:
                filename = "%s_%s_%s_%s.png" %(name, args.dataset_name, args.norm, args.target)
                load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out/', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)
            out_adv = net(adv_image)
            out_adv = torch.argmax(out_adv, dim=1)
            if args.target:
                success_num +=(out_adv == target_labels).sum()
            else:
                success_num +=(out_adv != labels).sum()
            
            
            if i % 20 == 0:
                num = i * batchsize
                adv_acc = success_num.item() / num

                print("Attack name: %s, dataset: %s, epoch: %d, asr: %.2f %%\n" %(name, args.dataset_name, i, adv_acc*100))

       
        total_num = len(test_loader.dataset)
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(name, args.dataset_name, success_num*100))

    if args.dataset_name == "cifar10":
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            adv_image= attack.forward(image, labels,None)
            
        
            if i==1:
                filename = "%s_%s_%s.png" %(name, args.dataset_name, args.norm)
                load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out/', filename)
                save_image(torch.cat([image, adv_image], 0),  load_path, nrow=args.batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)
            
            out = net(image)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            success_num +=(out_adv == labels).sum()

            if i % 1 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                adv_acc = 1 - adv_acc
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.2f %%\n" %(name, args.dataset_name, i, adv_acc*100))
        
        total_num = len(test_loader.dataset)
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        success_num = 1 - success_num
        
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(name, args.dataset_name, success_num*100))
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data preprocess args 
    parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--dataset_name', default='cifar10', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
    parser.add_argument('--overshoot', type= float, default=0.02)
    parser.add_argument('--max_iter', type= int, default=50)
    parser.add_argument('--cifar10_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/CIFAR10'), help='cifar10_path for this model')
    parser.add_argument('--imagenet_val_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/val.txt'), help='imagenet_val_path for this model')
    parser.add_argument('--imagenet_targrt_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/target.txt'), help='imagenet_targrt_path for this model')
    parser.add_argument('--imagenet_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/ILSVRC2012_img_val'), help='imagenet_path for this model')
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])

    args = parser.parse_args()

    test(args)
