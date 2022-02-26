import configargparse
import pdb

def pair(arg):
    return [float(x) for x in arg.split(',')]

def get_args():
    parser = configargparse.ArgParser(default_config_files=[])
    parser.add("--config", type=str, is_config_file=True, help="You can store all the config args in a config file and pass the path here")
    parser.add("--model_dir", type=str, default="models/model", help="Path to save/load the checkpoints, default=models/model")
    parser.add("--data_dir", type=str, default="datasets/", help="Path to load datasets from, default=datasets")
    parser.add("--dataset", "-d", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="Path to load dataset, default=cifar10")
    parser.add("--tf_seed", type=int, default=451760341, help="Random seed for initializing tensor-flow variables to rule out the effect of randomness in experiments, default=45160341") 
    parser.add("--np_seed", type=int, default=216105420, help="Random seed for initializing numpy variables to rule out the effect of randomness in experiments, default=216105420") 
    parser.add("--train_steps", type=int, default=80000, help="Maximum number of training steps, default=80000")
    parser.add("--out_steps", "-o", type=int, default=100, help="Number of output steps, default=100")
    parser.add("--summary_steps", type=int, default=500, help="Number of summary steps, default=500") 
    parser.add("--checkpoint_steps", "-c", type=int, default=1000, help="Number of checkpoint steps, default=1000")
    parser.add("--train_batch_size", "-b", type=int, default=128, help="The training batch size, default=128")
    parser.add("--step_size_schedule", nargs='+', type=pair, default=[[0, 0.1], [40000, 0.01], [60000, 0.001]], help="The step size scheduling, default=[[0, 0.1], [40000, 0.01], [60000, 0.001]], use like: --stepsize 0,0.1 40000,0.01 60000,0.001") 
    parser.add("--weight_decay", "-w", type=float, default=0.0002, help="The weight decay parameter, default=0.0002")
    parser.add("--momentum", type=float, default=0.9, help="The momentum parameter, default=0.9")
    parser.add("--replay_m", "-m", type=int, default=8, help="Number of steps to repeat trainig on the same batch, default=8")
    parser.add("--eval_examples", type=int, default=10000, help="Number of evaluation examples, default=10000")
    parser.add("--eval_size", type=int, default=128, help="Evaluation batch size, default=128")
    parser.add("--eval_cpu", type=bool, default=False, help="Set True to do evaluation on CPU instead of GPU, default=False")
    # params regarding attack
    parser.add("--epsilon", "-e", type=float, default=8.0, help="Epsilon (Lp Norm distance from the original image) for generating adversarial examples, default=8.0")
    parser.add("--pgd_steps", "-k", type=int, default=20, help="Number of steps to PGD attack, default=20")
    parser.add("--step_size", "-s", type=float, default=2.0, help="Step size in PGD attack for generating adversarial examples in each step, default=2.0")
    parser.add("--loss_func", "-f", type=str, default="xent", choices=["xent", "cw"], help="Loss function for the model, choices are [xent, cw], default=xent")
    parser.add("--num_restarts", type=int, default=1, help="Number of resets for the PGD attack, default=1")
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()

# TODO Default for model_dir
# TODO Need to update the helps
