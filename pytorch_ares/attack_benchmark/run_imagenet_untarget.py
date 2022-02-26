import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
os.environ['TORCH_HOME']=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'model/')
import time
import numpy as np
import timm
import third_party.efficientnet_pytorch
from torchvision import transforms
import torch
from pytorch_ares.attack_torch import *
from timm.utils import AverageMeter
import torch.nn.functional as F
from PIL import Image
import time
import math

class timm_model(torch.nn.Module):
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

class efficientnet_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.input_size = 224
        self.interpolation = 'bicubic'
        self.mean=(0.485,0.456,0.406)
        self.std=(0.229,0.224,0.225)
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean = torch.tensor(self.mean).view(3,1,1).to(self.device)
        self.std = torch.tensor(self.std).view(3,1,1).to(self.device)   
        x = (x - self.mean) / self.std
        labels = self.model(x.to(self.device))
        return labels

ATTACKS = {
    'fgsm': FGSM,
    'bim': BIM,
    'pgd': PGD,
    'mim': MIM,
    'cw': CW,
    'deepfool': DeepFool,
    'dim': DI2FGSM,
    'tim': TIFGSM,
    'si_ni_fgsm': SI_NI_FGSM,
    'vmi_fgsm': VMI_fgsm,
    'sgm': SGM,
    'cda': CDA,
    'tta': TTA
}

class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, meta_file, transform=None):

        self.data_dir = data_dir
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []
        for line in open(os.path.join(os.path.dirname(__file__), meta_file), encoding="utf-8"):
            img_path, label, target_label = line.strip().split(' ')
            self._indices.append((os.path.join(self.data_dir, img_path), label, target_label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label, target_label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        target_label=int(target_label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, target_label

def generate_attacker(args, net, device):
    if args.attack_name == 'fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'bim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'pgd':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    elif args.attack_name == 'mim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, stepsize=args.stepsize, steps=args.steps, decay_factor=args.decay_factor, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'cw':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, device,args.norm, args.target, args.kappa, args.lr, args.init_const, args.max_iter, 
                                args.binary_search_steps, args.dataset_name)
    elif args.attack_name == 'deepfool':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.overshoot, args.max_steps, args.norm, args.target, device)
    elif args.attack_name == 'dim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, 
                            resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'tim':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, p=args.norm, kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig, 
                            eps=args.eps, stepsize=args.stepsize, steps=args.steps, decay=args.decay_factor, resize_rate=args.resize_rate, 
                            diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)  
    elif args.attack_name == 'si_ni_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, scale_factor=args.scale_factor, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'vmi_fgsm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, epsilon=args.eps, p=args.norm, beta=args.beta, sample_number = args.sample_number, stepsize=args.stepsize, decay_factor=args.decay_factor, steps=args.steps, 
        data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name == 'sgm':
        attack_class = ATTACKS[args.attack_name]
        attack = attack_class(net, args.net_name, args.eps, args.norm, args.stepsize, args.steps, gamma=0.0, momentum=args.decay_factor, data_name=args.dataset_name,target=args.target, loss=args.loss, device=device)
    elif args.attack_name=='cda':
        attack_class=ATTACKS[args.attack_name]
        attack=attack_class(net, args.gk, p=args.norm, eps=args.eps, device=device)
    elif args.attack_name=='tta':
        attack_class=ATTACKS[args.attack_name]
        attack=attack_class(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps, kernel_size=5, nsig=3, resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, data_name=args.dataset_name,target=args.target, loss=args.loss,device=device)
    return attack

def get_pred_labels(args, input_tensor, model_name, device, gpu_id):
    if model_name.find('efficientnet')>=0:
        test_net=efficientnet_pytorch.EfficientNet.from_pretrained(model_name)
        test_net = efficientnet_model(device, test_net)
        test_input_size = test_net.input_size
        test_interpolation = test_net.interpolation
    else:
        test_net = timm.create_model(model_name, pretrained=True)
        test_net = timm_model(device, test_net)
        test_input_size = test_net.input_size
        test_interpolation = test_net.interpolation
                   
    test_net = test_net
    test_net.eval()
    resized_adv=F.interpolate(input=input_tensor, size=test_input_size, mode=test_interpolation)
    test_out=test_net(resized_adv)
    test_out = torch.argmax(test_out, dim=1)
    return test_out

def test(args):
    # target model
    device = torch.device(args.gpu)
    print('device: ')
    print(device)

    # initialize net and test loader
    net=None
    test_loader=None

    test_net = timm.create_model(args.target_name, pretrained=True)
    net = timm_model(device, test_net)
    input_size = net.input_size
    crop_pct = net.crop_pct
    interpolation = net.interpolation

    interpolation_mode={'nearest':0, 'bilinear':2, 'bicubic':3, 'box':4, 'hamming':5, 'lanczos':1}
    interpolation=interpolation_mode[interpolation]
    net.eval()

    # generate test loader
    input_resize=int(math.floor(input_size / crop_pct))
    val_transforms = [transforms.Resize(size=input_resize, interpolation=interpolation), transforms.CenterCrop(input_size), transforms.ToTensor()]
    val_dataset = ImageNetDataset(args.data_dir, args.label_file, transform=transforms.Compose(val_transforms))
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=1, 
        shuffle=False, pin_memory=True, drop_last=False)

    # initialize attacker
    attack=generate_attacker(args, net, device)

    # transfer models
    transfer_models=['vgg16', 'vgg19', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152',
        'densenet201', 'legacy_senet154', 'inception_v3', 'inception_v4', 'inception_resnet_v2', 
        'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 
        'efficientnet-b6', 'efficientnet-b7', 'vit_small_patch16_224', 'vit_base_patch16_224', 
        'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224']

    success_num = 0
    test_num= 0
    # true_classified_num=0
    
    averages={'clean': AverageMeter(), 'asr': AverageMeter()}
    for transfer_model in transfer_models:
        averages[transfer_model]=AverageMeter()

   
    for i, (image, labels, t_label) in enumerate(test_loader, 1):
        start_time=time.time()
        # generate adversary
        batchsize = image.shape[0]
        image, labels = image.to(device), labels.to(device)
        target_labels=None
        if args.target:
            target_labels=t_label.to(device)
        adv_image= attack.forward(image, labels, target_labels)
        
        with torch.no_grad():
            if not args.attack_name=='cda':
                # test clean acc and asr
                out = net(image)

                out_adv = net(adv_image)

                out_adv = torch.argmax(out_adv, dim=1)
                out = torch.argmax(out, dim=1)
                
                test_num += (out == labels).sum()
                if args.target:
                    success_num +=(out_adv == target_labels).sum()
                else:
                    success_num +=(out_adv != labels).sum()

                if i % 10 == 0:
                    num = i*batchsize
                    test_acc = test_num.item() / num
                    adv_acc = success_num.item() / num
                    print("Target model %s, Dataset %s, epoch %d, test acc %.2f %%" %(args.target_name, args.dataset_name, i, test_acc*100 ))
                    print("Attack name %s, dataset %s, epoch %d, asr %.2f %%\n" %(args.attack_name, args.dataset_name, i, adv_acc*100))

                # update averages
                averages['clean'].update((out == labels).sum().item()/batchsize, batchsize)
                if args.target:
                    averages['asr'].update((out_adv == target_labels).sum().item()/batchsize, batchsize)
                else:
                    averages['asr'].update((out_adv != labels).sum().item()/batchsize, batchsize)

            # test transfer models
            for transfer_model in transfer_models:
                transfer_pred=get_pred_labels(args, adv_image, transfer_model, device, args.gpu)
                
                if args.target:
                    averages[transfer_model].update((transfer_pred==target_labels).sum().item()/batchsize, batchsize)
                else:
                    averages[transfer_model].update((transfer_pred!=labels).sum().item()/batchsize, batchsize)
        end_time=time.time()
        print('Time of one epoch: %f' %(end_time-start_time))
    if not args.attack_name=='cda':
        total_num = len(test_loader.dataset)
    
        final_test_acc = test_num.item() / total_num
        success_num = success_num.item() / total_num
        print('Final: Target model %s, clean %.2f %%' %(args.target_name, averages['clean'].avg*100))
        print("Final: Attack %s, dataset %s, asr %.2f %%" %(args.attack_name, args.dataset_name, success_num*100))
        print("Final: Dataset %s, test acc %.2f %%" %(args.dataset_name, final_test_acc*100))
    
    for transfer_model in transfer_models:
        print('Final ASR: %s %.2f %%' %(transfer_model, averages[transfer_model].avg*100))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # data preprocess args 
    parser.add_argument("--gpu", type=str, default="cuda:0", help="Comma separated list of GPU ids")
    parser.add_argument('--dataset_name', default='imagenet', help= 'Dataset for this model', choices= ['cifar10', 'imagenet'])
    parser.add_argument('--crop_pct', type=float, default=0.875, help='Input image center crop percent') 
    parser.add_argument('--input_size', type=int, default=224, help='Input image size') 
    parser.add_argument('--interpolation', type=str, default='bilinear', choices=['bilinear', 'bicubic'], help='') 

    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support cw and deepfool', choices=[np.inf, 2])
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/val'), help= 'Dataset directory')
    parser.add_argument('--label_file', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/imagenet_val.txt'), help= 'Dataset directory')
    parser.add_argument('--net_name', default='tv_resnet50', help= 'net_name for sgm', choices= ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
    parser.add_argument('--attack_name', default='fgsm', help= 'Dataset for this model', choices= ['fgsm', 'bim', 'pgd','mim','si_ni_fgsm','vmi_fgsm','sgm', 'dim', 'tim', 'deepfool', 'cw','tta'])
    
    parser.add_argument('--target_name', default='inception_v3', help= 'target model', choices= ['resnet50', 'vgg16', 'inception_v3', 'swin_base_patch4_window7_224'])

    parser.add_argument('--eps', type= float, default=8/255, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/255/20, help='linf: 8/2550.0 and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=20, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--decay_factor', type= float, default=1.0, help='momentum is used')
    parser.add_argument('--resize_rate', type= float, default=0.85, help='dim is used')    #0.9
    parser.add_argument('--diversity_prob', type= float, default=0.7, help='dim is used')    #0.5
    parser.add_argument('--kernel_name', default='gaussian', help= 'kernel_name for tim', choices= ['gaussian', 'linear', 'uniform'])
    parser.add_argument('--len_kernel', type= int, default=15, help='len_kernel for tim')
    parser.add_argument('--nsig', type= int, default=3, help='nsig for tim')
    parser.add_argument('--scale_factor', type= int, default=1, help='scale_factor for si_ni_fgsm, min 1, max 5')
    parser.add_argument('--beta', type= float, default=1.5, help='beta for vmi_fgsm')
    parser.add_argument('--sample_number', type= int, default=10, help='sample_number for vmi_fgsm')
    
    parser.add_argument('--overshoot', type= float, default=0.02)
    parser.add_argument('--max_steps', type= int, default=50)

    parser.add_argument('--loss', default='ce', help= 'loss for fgsm, bim, pgd, mim, dim and tim', choices= ['ce', 'cw'])
    parser.add_argument('--kappa', type= float, default=0.0)
    parser.add_argument('--lr', type= float, default=0.2)
    parser.add_argument('--init_const', type= float, default=0.01)
    parser.add_argument('--binary_search_steps', type= int, default=4)
    parser.add_argument('--max_iter', type= int, default=200)
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])

    parser.add_argument('--ckpt_netG', type=str, default='attack_benchmark/checkpoints/netG_-1_img_incv3_imagenet_0_rl.pth', help='checkpoint path to netG of CDA')
    parser.add_argument('--gk', default=False, help= 'apply Gaussian smoothing to GAN output', choices= [True, False])
    args = parser.parse_args()

    test(args)

