# Learning-From-Noisy-Labels-on-Brain-MRI-Images
This repo contains the code for implementation of LNL methods CoTeaching and CoCorrecting on Brain MRI images for tumor detection


# Co-Correcting

Check out the paper here: [[paper](https://ieeexplore.ieee.org/document/9461766)][[arxiv](https://arxiv.org/abs/2109.05159)]

## Requirements:

+ python 3.11.11
+ numpy 2.2.1
+ pandas 2.2.3
+ scikit-learn 1.6.0
+ torch 2.5.1
+ torchvision 0.20.1


## Usage

`Co-Correcting.py` is used for both training a model on dataset with noisy labels and validating it.

Here is an example:

```shell
python Co-Correcting.py --dir ./experiment/ --dataset 'mnist' --noise_type sn --noise 0.2 --forget-rate 0.2 --epoch 320 --lambda1 0.3

python Co-Correcting.py --dataset 'mri' --noise_type sn --noise 0.2 --forget-rate 0.3 --epoch 100 --stage1 25 --stage2 70 --lambda1 0.3 --batch_size 16> output20.txt 2>&1

python Co-Correcting.py --dataset 'aptos' --noise_type sn --noise 0.2 --forget-rate 0.3 --epoch 100 --stage1 25 --stage2 70 --lambda1 0.3 --batch_size 16> output.txt 2>&1

```
keeping epoch numbers too low( <10) won't run on cpu
 
 forget rate > 0.4 is depreciating 
or you can train Co-Correcting with `.sh`:

```shell
sh script/mnist.sh
```
