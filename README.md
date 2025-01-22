# Learning-From-Noisy-Labels-on-Brain-MRI-Images
This repo contains the code for implementation of LNL methods CoTeaching and CoCorrecting on Brain MRI images for tumor detection


# Co-Correcting

Check out the paper here: [[paper](https://ieeexplore.ieee.org/document/9461766)][[arxiv](https://arxiv.org/abs/2109.05159)]

## Requirements:

+ python3.11
+ numpy
+ torch-
+ torchvision-

## Usage

`Co-Correcting.py` is used for both training a model on dataset with noisy labels and validating it.

Here is an example:

```shell
python Co-Correcting.py --dir ./experiment/ --dataRoot dataset/mnist --dataset 'mnist' --noise_type sn --noise 0.2 --forget-rate 0.2 --epoch 20
```
keeping epoch numbers too low( <10) won't run on cpu
 
or you can train Co-Correcting with `.sh`:

```shell
sh script/mnist.sh
```

