## How to set up datasets
By default, our implementation assumes that you have located all datasets in the `./datasets` folder. 
The implementation works for 
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), 
[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html), and 
 If you have already prepared the datasets on your machine, simply by copying the dataset folders into the `./datasets` and renaming the copied folder as one of the following names: `cifar10`, `cifar100`. 
Afterwards, just pass the name of the dataset as a parameter: 
```bash 
python train.py --dataset=DATASET_NAME
```

However, we find this way a bad practice, 
since for every implementation you will keep a copy of each dataset and that will waste the hard disk on machine. 
Alternatively, you may keep one folder on your machine for all the experiments that you have to avoid wasting your disk.
To set the directory path for the implementation to find your external data-sets folder, 
use the following `data_dir` argument: 
```bash 
python train.py --dataset=DATASET_NAME --data_dir=PATH_TO_YOUR_DATASET_DIR
```

The only drawback is the fact that you might be running many different experiments on your machine and you don't want to rename your CIFAR10 folder to `cifar10`, i.e. maybe for another experiment you need to name the file as something like `cifar10-dataset` or something similar, as a result, you might need to avoid renaming your dataset folder to `cifar10`. 
The final solution that we find very helpful is to create a symbolic link to your existing data-sets (Works for `UNIX` based OS e.g `MacOS` and `linux`)
This way, not only we avoid wasting disk on our machine, we also avoid renaming the folders.
For instance use the following commands:
```bash
ln -s PATH_TO_YOUR_CIFAR10_DATASET ./datasets/cifar10
```
See [tutorail](https://kb.iu.edu/d/abbe "ln tutorail") for more info about Symbolic Links. 
