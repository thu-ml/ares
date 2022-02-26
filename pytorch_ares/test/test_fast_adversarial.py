import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import numpy as np
from pytorch_ares.dataset_torch.cifar_dataset import cifar10
from third_party.attack_cifar import *
from third_party.autoattack.autoattack import AutoAttack
from pytorch_ares.cifar10_model.utils import load_model_from_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="Comma separated list of GPU ids")
    parser.add_argument('--result_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out'), help='result path for cifar10')
    #dataset 
    parser.add_argument('--batchsize', default=10, help= 'batchsize for this model')
    parser.add_argument('--cifar10_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/CIFAR10'), help='cifar10_path for this model')
    
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--version', default='rand', help= 'version for autoattack', choices= ['standard', 'plus', 'rand'])
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    args = parser.parse_args()
    
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'pytorch_ares/cifar10_model/fast_at.py')
    rs_model = load_model_from_path(model_path)
    net = rs_model.load(device)

    test_loader = cifar10(args.batchsize, args.cifar10_path)
    test_num = 0
    test_num_pgd = 0
    test_num_apgd = 0
    test_num_square = 0
    test_num_aa = 0
    with open(os.path.join(args.result_path, "{}.csv".format("awp")), "a") as f:
        for i, (image,labels) in enumerate(test_loader, 1):
            batchsize = image.shape[0]
            image, labels = image.to(device), labels.to(device)
            out = net(image)
            out = torch.argmax(out, dim=1)
            acc = (out == labels)
            test_num += (out == labels).sum()
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0) 
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = image[ind_to_fool].clone(), labels[ind_to_fool].clone()
            
                Pgd = PGD(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps,target=args.target, device=device)
                autoattck = AutoAttack(net,norm="Linf",steps=args.steps, query=args.n_queries, eps=args.eps, version=args.version,device=device)

                
                adv_pgd= Pgd.forward(x_to_fool, y_to_fool)
                adv_autoattack = autoattck.run_standard_evaluation(x_to_fool, y_to_fool,bs=len(y_to_fool))
                
                # attack output
                out_adv_pgd = net(adv_pgd)
                out_adv_autoattack = net(adv_autoattack)
            
            
                out_adv_pgd = torch.argmax(out_adv_pgd, dim=1)
                out_adv_autoattack = torch.argmax(out_adv_autoattack, dim=1)
                
                test_num_pgd += (out_adv_pgd == y_to_fool).sum()
                test_num_aa += (out_adv_autoattack == y_to_fool).sum()

                if i % 50 == 0:
                    num = i*batchsize
                    test_acc = test_num.item() / num
                    
                    test_acc_pgd = test_num_pgd.item() / num
                    test_acc_aa = test_num_aa.item() / num
                    
                    print("epoch: %d clean_acc: %.2f %%" %(i, test_acc*100 ))
                    print("epoch: %d pgd_acc: %.2f %%" %(i, test_acc_pgd*100))
                    print("epoch: %d autoattack_acc: %.2f %%\n" %(i, test_acc_aa*100))
        
        total_num = len(test_loader.dataset)
        final_test_acc = test_num.item() / total_num
        
        success_num_pgd = test_num_pgd.item() / total_num
        success_num_aa = test_num_aa.item() / total_num
    
        print("clean_acc: %.2f %%" %(final_test_acc*100))
        print("pgd_acc: %.2f %%" %(success_num_pgd*100))
        print("autoattack_acc: %.2f %%\n" %(success_num_aa*100))
        f.write(f"clean_acc: %.2f %%" %(final_test_acc*100))
        f.write(f"pgd_acc: %.2f %%\n" %(success_num_pgd*100))
        f.write(f"autoattack_acc: %.2f %%\n" %(success_num_aa*100))
        
