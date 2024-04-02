# BayesianSpikeFusion
This repository provides an implementation of BayesianSpikeFusion to improve the trade-off between energy consumption and accuracy of SNNs.

To replicate the experiments in the main text, please follow the steps below:
1. [Prepare environment](#1-prepare-environment)
2. [Training](#2-training)
3. [ANN-SNN conversion](#3-ann-snn-conversion)
4. [Tuning Hyperparameters](#4-tuning-hyperparameters)
5. [Simulate SNN](#5-simulate-snn)
6. [Estimate with BayesianSpikeFusion](#6-estimate-with-bayesianspikefusion)

## 1. Prepare environment
First, set up the environment to execute the program. The scripts in this repository are designed to be run with Python 3.8.13. Please install the libraries using the requirements.txt file as shown in the command below:

```sh
pip install -r requirements.txt
```

Additionally, if using TinyImagenet, manual setup is required. Please download and organize TinyImagenet using [this Gist](https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4).

## 2. Training
To train an Artificial Neural Network (ANN), you will use the ```train_ann.py``` script. The dataset name and model name are mandatory arguments, and you can specify the location to insert the intermediate output layer using the ```--ic_index``` argument. You can also specify the directory to save the dataset with the ```--data_path``` argument, and the directory to save the experimental results with the ```--root``` argument. 

Here's an example command for training a VGG19 model with an intermediate output layer added after the 6th layer, using the CIFAR-10 dataset:
```sh
python train.py cifar10 vgg19 --ic_index 6 --data_path /path/to/dataset/directory --root /path/to/experimental/results
```
Make sure to replace ```/path/to/dataset/directory``` and ```/path/to/experimental/results``` with the actual paths to your dataset directory and experimental results directory, respectively.

## 3. ANN-SNN conversion
To convert the trained ANN to SNN, use the `ann2snn.py` file. The first argument of this script is mandatory, specifying the directory where the trained ANN model is saved. The ANN-SNN converted SNN model will be saved in this directory.

In our experiments, we adopt Robust Normalization as the ANN-SNN conversion method, converting weights and biases so that the n percentile value of the activation values of ANN matches the threshold of SNN. You can specify this percentile value n with the `--percentile` argument, which is set to 0.999 by default.

```sh
python ann2snn.py /path/to/trained_model_saved --percentile 0.999
```

## 4. Tuning Hyperparameters
Next, tune the hyperparameters of BayesianSpikeFusion using `tune_hyperparams.py`. Similar to ANN-SNN conversion, the first argument is mandatory, specifying the directory where the trained ANN model or converted SNN model is saved. The results of searching for the hyperparameter α of BayesianSpikeFusion using Grid Search and the results of searching using empirical Bayesian method are saved in `.npz` format in this directory.

```sh
python tune_hyperparams.py /path/to/trained_model_saved
```

## 5. Simulate SNN
SNN can be simulated using the `simulate_snn.py` file. Similar to previous files, the first argument is mandatory, specifying the directory where the trained ANN model or converted SNN model is saved. Under this directory, create `snn_xxx_yyy/` directories and save the output spike sequences and teacher labels in batches in npz format.

```sh
python simulate_snn.py /path/to/trained_model_saved
```

## 6. Estimate with BayesianSpikeFusion
Finally, obtain the inference results of SNN with BayesianSpikeFusion from the data of the hyperparameter α of BayesianSpikeFusion and the saved output spike sequences. Use the `plot.py` file not only to calculate the results of BayesianSpikeFusion but also to plot the graph of energy-accuracy. The plot results are saved in the `Plot/` directory directly under this repository.

When executing `plot.py`, you can specify multiple SNN results from which to calculate the results of BayesianSpikeFusion. This is a mandatory argument, achievable by entering the directory saved in step 5.

```sh
python plot.py {/path/to/trained_model_saved}/snn_xxx_yyy 
```
