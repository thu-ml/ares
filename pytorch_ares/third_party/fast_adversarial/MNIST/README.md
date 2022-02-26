# fast_is_better_than_free_MNIST

To train, run 

`python train_mnist.py --fname models/fgsm.pth` 

which runs FGSM training with the default parameters. To run the evaluation with default parameters (50 iterations with step size 0.01 and 10 random restarts), run 

`python evaluate_mnist.py --fname models/fgsm.pth`

To run PGD adversarial training with the same parameters as those used [here](https://github.com/MadryLab/mnist_challenge/blob/master/config.json), run 

`python train_mnist.py --fname models/pgd_madry.pth --attack pgd --alpha 0.01 --lr-type flat --lr-max 0.0001 --epochs 100 --batch-size 50`