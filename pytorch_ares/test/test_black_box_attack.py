import numpy as np
import timm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['TORCH_HOME']=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'model/')
from torchvision.utils import save_image
import torch
from third_party.example.cifar10.pytorch_cifar10.models import *
from pytorch_ares.dataset_torch.datasets_test import datasets
from pytorch_ares.attack_torch import *
ATTACKS = {
    'ba': BoundaryAttack,
    'spsa': SPSA,   
    'nes': NES,
    'nattack':Nattack,
    'evolutionary':Evolutionary,
}

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
    distortion = 0
    dist= 0
    success_num = 0
    test_num= 0
    
    if args.attack_name == 'ba':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.spherical_step_eps, args.norm,args.orth_step_factor,args.perp_step_factor, 
                        args.orthogonal_step_eps, args.max_queries, args.dataset_name, device,args.target)
    if args.attack_name == 'spsa':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,norm=args.norm, device=device, eps=args.eps, learning_rate=args.learning_rate, delta=args.delta, spsa_samples=args.spsa_samples, 
                 sample_per_draw=args.sample_per_draw, nb_iter=args.max_queries, data_name=args.dataset_name,early_stop_loss_threshold=None, IsTargeted=args.target)
    if args.attack_name == 'nes':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, nes_samples=args.nes_samples, sample_per_draw=args.nes_per_draw, 
                              p=args.norm, max_queries=args.max_queries, epsilon=args.epsilon, step_size=args.stepsize,
                device=device, data_name=args.dataset_name, search_sigma=0.02, decay=1.0, random_perturb_start=True, target=args.target)
    if args.attack_name == 'nattack':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, eps=args.epsilon, max_queries=args.max_queries, device=device,data_name=args.dataset_name, 
                              distance_metric=args.norm, target=args.target, sample_size=args.sample_size, lr=args.lr, sigma=args.sigma)
    if args.attack_name == 'evolutionary':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net,args.dataset_name,args.target,device, args.ccov, 
        args.decay_weight, args.max_queries, args.mu, args.sigma, args.maxlen)


    if args.dataset_name == "imagenet":
        for i, (image,labels,target_labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            target_labels = target_labels.to(device)
            adv_image= attack.forward(image, labels, target_labels)
            distortion1 = torch.mean((adv_image-image)**2) / ((1-0)**2)
            distortion +=distortion1
            if args.norm== 2:
                dist1 = torch.norm(adv_image - image, p=2)
                dist+=dist1

            elif args.norm==np.inf:
                dist1 = torch.max(torch.abs(adv_image - image)).item()
                dist+=dist1
            if i ==1:
                filename = "%s_%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm, args.target)
                load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out/', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            out = net(image)
            #print(out_adv.shape)
            out_adv = net(adv_image)
            
            out_adv = torch.argmax(out_adv, dim=1)
            out = torch.argmax(out, dim=1)
            
            test_num += (out == labels).sum()
            if args.target:
                success_num +=(out_adv == target_labels).sum()
            else:
                success_num +=(out_adv != labels).sum()
            
            if i % 2 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                db_mean_distance = dist / num
                distortion_mean = distortion / num
                print("%s数据集第%d次分类准确率：%.4f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.4f %%" %(args.attack_name, args.dataset_name, i, adv_acc*100))
                print("%s在%s数据集第%d次平均距离：%f" %(args.attack_name, args.dataset_name, i, db_mean_distance))
                print("%s在%s数据集第%d次平均失真：%e \n" %(args.attack_name, args.dataset_name, i, distortion_mean))
       
        total_num = len(test_loader.dataset)
      
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        db_mean_distance = dist / total_num
        distortion_mean = distortion / total_num
        
        print("%s数据集分类准确率：%.4f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.4f %%" %(args.attack_name, args.dataset_name, success_num*100))
        print("%s在%s数据集的平均距离：%.5f" %(args.attack_name, args.dataset_name, db_mean_distance))
        print("%s在%s数据集的平均失真：%e \n" %(args.attack_name, args.dataset_name, distortion_mean))

    if args.dataset_name == "cifar10":
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            out = net(image)
            out = torch.argmax(out, dim=1)
           
            adv_image= attack.forward(image, labels, None)
            distortion1 = torch.mean((adv_image-image)**2) / ((1-0)**2)
            distortion +=distortion1
            if args.norm==2:
                dist1 = torch.norm(adv_image - image, p=2)
                dist+=dist1
            elif args.norm==np.inf:
                dist1 = torch.max(torch.abs(adv_image - image)).item()
                dist+=dist1

        
            if i==1:
                filename = "%s_%s_%s.png" %(args.attack_name, args.dataset_name, args.norm)
                load_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out/', filename)
                save_image( torch.cat([image, adv_image], 0),  load_path, nrow=batchsize, padding=2, normalize=True, 
                            range=(0,1), scale_each=False, pad_value=0)

            
            #print(out_adv.shape)
            out_adv = net(adv_image)
        
            out_adv = torch.argmax(out_adv, dim=1)
            success_num +=(out_adv != labels).sum()
            
            test_num += (out == labels).sum()

            if i % 1 == 0:
                num = i*batchsize
                test_acc = test_num.item() / num
                adv_acc = success_num.item() / num
                db_mean_distance = dist / num
                distortion_mean = distortion / num
                print("%s数据集第%d次分类准确率：%.2f %%" %(args.dataset_name, i, test_acc*100 ))
                print("%s在%s数据集第%d次攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, i, adv_acc*100))
                print("%s在%s数据集第%d次的平均距离：%f" %(args.attack_name, args.dataset_name, i, db_mean_distance))
                print("%s在%s数据集第%d次的平均失真：%e \n" %(args.attack_name, args.dataset_name, i,  distortion_mean))
        total_num = len(test_loader.dataset)
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / num
        db_mean_distance = dist / num
        distortion_mean = distortion / num
        print("%s数据集分类准确率：%.2f %%" %(args.dataset_name, final_test_acc*100))
        print("%s在%s数据集攻击成功率：%.2f %%" %(args.attack_name, args.dataset_name, success_num*100))
        print("%s在%s数据集的平均距离：%f" %(args.attack_name, args.dataset_name, db_mean_distance))
        print("%s在%s数据集的平均失真：%e \n" %(args.attack_name, args.dataset_name, distortion_mean))
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data preprocess args 
    parser.add_argument("--gpu", type=str, default="2", help="Comma separated list of GPU ids")
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='')
    parser.add_argument('--dataset_name', default='cifar10', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--norm', default= 2, help='You can choose linf and l2', choices=[np.inf, 1, 2])
    parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
    parser.add_argument('--cifar10_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/CIFAR10'), help='cifar10_path for this model')
    parser.add_argument('--imagenet_val_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/val.txt'), help='imagenet_val_path for this model')
    parser.add_argument('--imagenet_targrt_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/target.txt'), help='imagenet_targrt_path for this model')
    parser.add_argument('--imagenet_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/ILSVRC2012_img_val'), help='imagenet_path for this model')
    parser.add_argument('--attack_name', default='evolutionary', help= 'Dataset for this model', choices= ['ba','spsa', 'nes','nattack','evolutionary'])
    #boundary
    parser.add_argument('--spherical_step_eps', type= float, default=1e-2)
    parser.add_argument('--orthogonal_step_eps', type= float, default=1e-2)
    parser.add_argument('--orth_step_factor', type= float, default=0.97)
    parser.add_argument('--perp_step_factor', type= float, default=0.97)
    parser.add_argument('--max_queries', type= int, default=20000, help='max_queries for black-box attack based on queries')
    #spsa
    parser.add_argument('--eps', type= float, default= 16/255.0, help='eps for spsa, 0.05 for linf 3.0')
    parser.add_argument('--learning_rate', type= float, default=0.006, help='learning_rate for spsa')
    parser.add_argument('--delta', type= float, default=1e-3, help='delta for spsa')
    parser.add_argument('--spsa_samples', type= int, default= 10, help='spsa_samples for spsa')
    parser.add_argument('--sample_per_draw', type= int, default=20, help='spsa_iters for spsa')
    #nes
    parser.add_argument('--epsilon', type= float, default= 16/255.0, help='eps for spsa, 0.05 for linf')
    parser.add_argument('--stepsize', type= float, default=16/25500.0, help='learning_rate for spsa')
    parser.add_argument('--max_iter', type= int, default=100, help='max_iter for spsa')
    parser.add_argument('--nes_samples', default= 10, help='nes_samples for nes')
    parser.add_argument('--nes_per_draw', type= int, default=20, help='nes_iters for nes')
    #nattack
    parser.add_argument('--sample_size', type= int, default=100, help='sample_size for nattack')
    parser.add_argument('--lr', type= float, default= 0.02, help='lr for nattack')
    parser.add_argument('--sigma', type= float, default= 0.1, help='sigma for nattack')
    #Evolutionary
    parser.add_argument('--ccov', type= float, default= 0.001, help='eps for spsa, 0.05 for linf')
    parser.add_argument('--decay_weight', type= float, default=0.99, help='learning_rate for spsa')
    parser.add_argument('--mu', type= float, default=1e-2, help='max_iter for spsa')
    parser.add_argument('--sigmaa', type= float, default= 3e-2, help='nes_samples for nes')
    parser.add_argument('--maxlen', type= int, default=30, help='nes_iters for nes')
    
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    args = parser.parse_args()

    test(args)
