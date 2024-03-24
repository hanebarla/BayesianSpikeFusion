# BayesianSpikeFusion

## 1. Prepare environment
First, set up the environment to execute the program. The scripts in this repository are designed to be run with Python 3.8.13. Please install the libraries using the requirements.txt file as shown in the command below:

```sh
pip install -r requirements.txt
```

加えて，TinyImagenetを使用する場合は手動でセットアップする必要があります．
こちらからTinyImagenetのダウンロードとディレクトリ整理を行ってください．

## 2. Training
To train an Artificial Neural Network (ANN), you will use the ```train_ann.py``` script. The dataset name and model name are mandatory arguments, and you can specify the location to insert the intermediate output layer using the ```--ic_index``` argument. You can also specify the directory to save the dataset with the ```--data_path``` argument, and the directory to save the experimental results with the ```--root``` argument.

Here's an example command for training a VGG19 model with an intermediate output layer added after the 6th layer, using the CIFAR-10 dataset:
```sh
python train.py cifar10 vgg19 --ic_index 6 --data_path /path/to/dataset/directory --root /path/to/experimental/results
```
Make sure to replace ```/path/to/dataset/directory``` and ```/path/to/experimental/results``` with the actual paths to your dataset directory and experimental results directory, respectively.

## 3. ANN-SNN conversion

```sh
python ann2snn.py
```

## 4. Tuning Hyperparameters


## 5. Simulate SNN with BSF


