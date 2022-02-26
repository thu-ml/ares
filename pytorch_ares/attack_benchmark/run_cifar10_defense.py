import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import csv
import time
import json
import numpy as np
from pytorch_ares.dataset_torch.cifar_dataset import cifar10
from third_party.attack_cifar import *
from third_party.autoattack.autoattack import AutoAttack
from pytorch_ares.cifar10_model.utils import load_model_from_path
model_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
models = [os.path.join(model_path, 'pytorch_ares/cifar10_model/awp.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/fast_at.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/featurescatter.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/trades.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/at_he.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/hydra.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/label_smoothing.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/pre_training.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/robust_overfiting.py'),
          os.path.join(model_path, 'pytorch_ares/cifar10_model/rst.py')]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2", help="Comma separated list of GPU ids")
    parser.add_argument('--result_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'test_out'), help='result path for cifar10')
    #dataset 
    parser.add_argument('--batchsize', default=250, help= 'batchsize for this model')
    parser.add_argument('--cifar10_path', default=os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'data/CIFAR10'), help='cifar10_path for this model')
    
    parser.add_argument('--norm', default=np.inf, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--norm_apgd', default=2, help='You can choose np.inf and 2(l2), l2 support all methods and linf dont support apgd and deepfool', choices=[np.inf, 2])
    parser.add_argument('--eps', type= float, default=8/255.0, help='linf: 8/255.0 and l2: 3.0')
    parser.add_argument('--stepsize', type= float, default=8/24000.0, help='linf: eps/steps and l2: (2.5*eps)/steps that is 0.075')
    parser.add_argument('--steps', type= int, default=100, help='linf: 100 and l2: 100, steps is set to 100 if attack is apgd')
    parser.add_argument('--n_queries', default= 5000, help= 'n_queries for square')
    parser.add_argument('--version', default='standard', help= 'version for autoattack', choices= ['standard', 'plus', 'rand'])
    parser.add_argument('--target', default=False, help= 'target for attack', choices= [True, False])
    args = parser.parse_args()
    
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    device = torch.device(f"cuda:{gpu_list[0]}" if torch.cuda.is_available() else "cpu")
    

    test_loader = cifar10(args.batchsize, args.cifar10_path)
    
    with open(os.path.join(args.result_path, "{}.csv".format("result")), "w") as f:
        result = []
        name = "clean"
        robust = "robust"
        for model_path in models:
            test_num = 0
            robust_num = 0
            test_num_pgd = 0
            num_miss = 0
            test_num_aa = 0
            false_class = 0
            print('Loading {}...'.format(model_path))
            rs_model = load_model_from_path(model_path)
            net = rs_model.load(device)
            for i, (image,labels) in enumerate(test_loader, 1):
                torch.cuda.synchronize()
                start_time = time.time()
                batchsize = image.shape[0]
                image, labels = image.to(device), labels.to(device)
                Pgd = PGD(net, epsilon=args.eps, norm=args.norm, stepsize=args.stepsize, steps=args.steps,target=args.target, device=device)
                pgd_name = Pgd.__class__.__name__
                autoattck = AutoAttack(net,norm="Linf",steps=args.steps, query=args.n_queries, eps=args.eps, version=args.version,device=device)
                aa_name = autoattck.__class__.__name__

                out = net(image)
                out = torch.argmax(out, dim=1)
                acc = (out == labels)
                test_num += (out == labels).sum()
                
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0) 
                if ind_to_fool.numel() != 0:
                    x_to_fool, y_to_fool = image[ind_to_fool].clone(), labels[ind_to_fool].clone()
                    false_class += len(y_to_fool)
                    k = batchsize - len(y_to_fool)
                    num_miss = k
                    
                    
                    adv_autoattack = autoattck.run_standard_evaluation(x_to_fool, y_to_fool,bs=len(y_to_fool))
                    out_adv_autoattack = net(adv_autoattack)
                    out_adv_autoattack = torch.argmax(out_adv_autoattack, dim=1)
                    acc_aa = (out_adv_autoattack != y_to_fool)
                    test_num_aa += (out_adv_autoattack == y_to_fool).sum()
                    # attack output
                    adv_pgd= Pgd.forward(x_to_fool, y_to_fool)
                    out_adv_pgd = net(adv_pgd)
                    out_adv_pgd = torch.argmax(out_adv_pgd, dim=1)
                    acc_pgd = (out_adv_pgd != y_to_fool)
                      
                    robust_num += (acc_pgd | acc_aa).nonzero().numel()+ num_miss
                    test_num_pgd += (out_adv_pgd == y_to_fool).sum()
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time
                    print("The time elapse of epoch {:05d}".format(i) + " is: " +
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))


                    if i % 1 == 0:
                        num = i*batchsize
                        test_acc = test_num.item() / num
                        print("epoch: %d %s clean_acc: %.2f %%" %(i, net.name, test_acc*100 ))
                        
                        if false_class!=0:
                            test_acc_pgd = test_num_pgd.item() / num
                            test_acc_aa = test_num_aa.item() / num
                            
                            print("epoch: %d %s pgd_acc: %.2f %%" %(i, net.name, test_acc_pgd*100))
                            print("epoch: %d %s autoattack_acc: %.2f %%\n" %(i, net.name, test_acc_aa*100))
            
            total_num = len(test_loader.dataset)
            print(total_num)
            final_test_acc = test_num.item() / total_num
            print(test_num.item())
            robust_acc = robust_num / total_num
            robust_acc = 1 - robust_acc
            
            success_num_pgd = test_num_pgd.item() / total_num
            success_num_aa = test_num_aa.item() / total_num
            
            print("%s clean_acc: %.2f %%" %(net.name, round(final_test_acc*100,2)))
            print("%s pgd_acc: %.2f %%" %(net.name, round(success_num_pgd*100,2)))
            print("%s autoattack_acc: %.2f %%\n" %(net.name, round(success_num_aa*100,2)))
            print("%s robust_acc: %.2f %%\n" %(net.name, round(robust_acc*100,2)))
            f.write("{},{},{}\n".format(name, net.name, round(final_test_acc*100,2)))
            f.write("{},{},{}\n".format(pgd_name, net.name, round(success_num_pgd*100,2)))
            f.write("{},{},{}\n".format(aa_name, net.name, round(success_num_aa*100,2)))
            f.write("{},{},{}\n".format(robust, net.name, round(robust_acc*100,2)))


    with open(os.path.join(args.result_path, "{}.csv".format("result")), "r") as f:
        total_score = 0
        total_num = 0
        reader = csv.reader(f)
        result = []
        for attack_name, model, score in reader:
            total_score += float(score)
            total_num += 1
            result.append([attack_name, model, score])

        avg_score = round(total_score / total_num, 2)
        result.append(["--", "--", avg_score])
        print("ResultScores: %s" % json.dumps(result))