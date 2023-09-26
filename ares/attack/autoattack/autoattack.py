import math
import time
import numpy as np
import torch
from ares.attack.autoattack import checks
from ares.utils.registry import registry

@registry.register_attack('autoattack')
class AutoAttack():
    '''A class to perform autoattack. It is called by registry.

    Example:
        >>> from ares.utils.registry import registry
        >>> attacker_cls = registry.get_attack('autoattack')
    '''

    def __init__(self, model, device='cuda', norm=np.inf, eps=.3, seed=None, verbose=False,
                 attacks_to_run=[], version='standard', is_tf_model=False,
                 logger=None):
        '''

        Args:
            model (torch.nn.Module): The target model to be attacked.
            device (torch.device): The device to perform autoattack. Defaults to 'cuda'.
            norm (float): The norm of distance calculation for adversarial constraint.
                Defaults to np.inf. It is selected from [1, 2, np.inf].
            eps (float): The maximum perturbation range epsilon.
            seed (float): Random seed. Defaults to None.
            verbose (bool): Output the details during the attack process. Defaults to True.
            attacks_to_run (list): Set the attacks to run. Defaults to []. It should be selected
                from ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t'].
            version (str): Define the version of attack. Defaults to 'standard'. It is selected
                from ['standard', 'plus', 'rand'].
            is_tf_model (bool): Whether the model is based on tensorflow. Defaults to False.
            log_path (str): Path to the log file. Defaults to None.
        '''

        self.model = model
        self.norm = None
        assert norm in [1, 2, np.inf]
        if norm == 1:
            self.norm = 'L1'
        elif norm == 2:
            self.norm = 'L2'
        else:
            self.norm = 'Linf'
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.is_tf_model = is_tf_model
        self.device = device
        self.logger = logger
        if self.verbose:
            assert self.logger is not None, "Must set logger, if verbose is True."
        
        assert not self.is_tf_model, "Only pytorch models supported."

        if version in ['standard', 'plus', 'rand'] and attacks_to_run != []:
            raise ValueError("attacks_to_run will be overridden unless you use version='custom'")
        
        from .autopgd_base import APGDAttack
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
            device=self.device, logger=self.logger)
        
        from .fab_pt import FABAttack_PT
        self.fab = FABAttack_PT(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False, device=self.device)
    
        from .square import SquareAttack
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
            
        from .autopgd_base import APGDAttack_targeted
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device,
            logger=self.logger)
        
        if version in ['standard', 'plus', 'rand']:
            self.set_version(version)
        
    def get_logits(self, x):
        '''This function calculates the logits of the target model.'''
        
        return self.model(x)
    
    def get_seed(self):
        '''This function automatically set a random seed.'''

        return time.time() if self.seed is None else self.seed
    
    def __call__(self, images=None, labels=None, target_labels=None):
        '''This function perform attack on target images with corresponding labels.

        Args:
            images (torch.Tensor): The images to be attacked. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].
            labels (torch.Tensor): The corresponding labels of the images. The labels should be torch.Tensor with shape [N, ]
            target_labels (torch.Tensor): Not used in autoattack and should be None type.
        '''
        assert target_labels is None, "Target attack is not necessary for autoattack."
        x_adv = self.run_standard_evaluation(images, labels, bs=images.size(0), return_labels=False)
    
    def run_standard_evaluation(self, images, labels, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        # checks on type of defense
        if self.version != 'rand':
            checks.check_randomized(self.get_logits, images[:bs].to(self.device),
                labels[:bs].to(self.device), bs=bs, logger=self.logger)
        n_cls = checks.check_range_output(self.get_logits, images[:bs].to(self.device),
            logger=self.logger)
        checks.check_dynamic(self.model, images[:bs].to(self.device), self.is_tf_model,
            logger=self.logger)
        checks.check_n_classes(n_cls, self.attacks_to_run, self.apgd_targeted.n_target_classes,
            self.fab.n_target_classes, logger=self.logger)
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(images.shape[0] / bs))
            robust_flags = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
            y_adv = torch.empty_like(labels)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, images.shape[0])

                x = images[start_idx:end_idx, :].clone().to(self.device)
                y = labels[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x).max(dim=1)[1]
                y_adv[start_idx: end_idx] = output
                correct_batch = y.eq(output)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)

            robust_accuracy = torch.sum(robust_flags).item() / images.shape[0]
            robust_accuracy_dict = {'clean': robust_accuracy}
            
            if self.verbose:
                self.logger.info('initial accuracy: {:.2%}'.format(robust_accuracy))
                    
            x_adv = images.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                # item() is super important as pytorch int division uses floor rounding
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))

                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = images[batch_datapoint_idcs, :].clone().to(self.device)
                    y = labels[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd-ce':
                        # apgd on cross-entropy loss
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'apgd-dlr':
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab':
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    elif attack == 'square':
                        # square
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)
                    
                    elif attack == 'apgd-t':
                        # targeted apgd
                        self.apgd_targeted.seed = self.get_seed()
                        adv_curr = self.apgd_targeted.perturb(x, y) #cheap=True
                    
                    elif attack == 'fab-t':
                        # fab targeted
                        self.fab.targeted = True
                        self.fab.n_restarts = 1
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    else:
                        raise ValueError('Attack not supported')
                
                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        self.logger.info('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                
                robust_accuracy = torch.sum(robust_flags).item() / images.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                if self.verbose:
                    self.logger.info('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))
                    
            # check about square
            checks.check_square_sr(robust_accuracy_dict, logger=self.logger)
            
            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - images).abs().reshape(images.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - images) ** 2).reshape(images.shape[0], -1).sum(-1).sqrt()
                elif self.norm == 'L1':
                    res = (x_adv - images).abs().reshape(images.shape[0], -1).sum(dim=-1)
                self.logger.info('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.info('robust accuracy: {:.2%}'.format(robust_accuracy))
        if return_labels:
            return x_adv, y_adv
        else:
            return x_adv
        
    def clean_accuracy(self, images, labels, bs=250):
        n_batches = math.ceil(images.shape[0] / bs)
        acc = 0.
        for counter in range(n_batches):
            x = images[counter * bs:min((counter + 1) * bs, images.shape[0])].clone().to(self.device)
            y = labels[counter * bs:min((counter + 1) * bs, images.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / images.shape[0]))
        
        return acc.item() / images.shape[0]
        
    def run_standard_evaluation_individual(self, images, labels, bs=250, return_labels=False):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                ', '.join(self.attacks_to_run)))
        
        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            x_adv, y_adv = self.run_standard_evaluation(images, labels, bs=bs, return_labels=True)
            if return_labels:
                adv[c] = (x_adv, y_adv)
            else:
                adv[c] = x_adv
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(x_adv, labels, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.info('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv
        
    def set_version(self, version='standard'):
        '''The function to set the attack version.

        Args:
            version (str): The version of attack. Defaults to 'standard'.
        '''

        if self.verbose:
            print('setting parameters for {} version'.format(version))
        
        if version == 'standard':
            self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            if self.norm in ['Linf', 'L2']:
                self.apgd.n_restarts = 1
                self.apgd_targeted.n_target_classes = 9
            elif self.norm in ['L1']:
                self.apgd.use_largereps = True
                self.apgd_targeted.use_largereps = True
                self.apgd.n_restarts = 5
                self.apgd_targeted.n_target_classes = 5
            self.fab.n_restarts = 1
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            #self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
        
        elif version == 'plus':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
            self.apgd.n_restarts = 5
            self.fab.n_restarts = 5
            self.apgd_targeted.n_restarts = 1
            self.fab.n_target_classes = 9
            self.apgd_targeted.n_target_classes = 9
            self.square.n_queries = 5000
            if not self.norm in ['Linf', 'L2']:
                print('"{}" version is used with {} norm: please check'.format(
                    version, self.norm))
        
        elif version == 'rand':
            self.attacks_to_run = ['apgd-ce', 'apgd-dlr']
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20

