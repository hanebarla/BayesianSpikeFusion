import os, argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, ConstantKernel
from scipy import interpolate
from scipy.special import beta as Beta
from tqdm import trange

from argument import get_snn_args, load_args, get_save_snn_dir
from logger import create_logger
from dataset import dataloader_factory
from model_ann import model_factory
from model_snn import create_model_snn
from ann2snn import convert

def main():
    snn_args = get_snn_args()
    args = load_args(os.path.join(snn_args.train_dir, "command.json"))
    args.snn = True
    args.batch_size = snn_args.batch_size

    conditions, _ = get_save_snn_dir(args)

    # create logger
    logger = create_logger(snn_args.train_dir, conditions, 'tune_{}_log.txt'.format(snn_args.hps))
    logger.info("[ANN Args]: {}".format(str(args.__dict__)))
    logger.info("[SNN Args]: {}".format(str(snn_args.__dict__)))

    # get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cnt = torch.cuda.device_count()
    logger.info("[Device]: {}".format(device))
    logger.info("[Device Count]: {}".format(device_cnt))

    # get Dataloader
    train_dataloader, val_dataloader, test_dataloader, num_classes, in_shapes, _ = dataloader_factory(args)
    logger.info("Mixup: {}".format(False))

    # get ANN model
    ann_model = model_factory(args, num_classes, in_shapes)
    logger.info("[model]: {}".format(str(ann_model)))
    checkpoint = torch.load(os.path.join(snn_args.train_dir, "checkpoint.pth"))
    logger.info("ANN Accuracy: {}".format(checkpoint["acc"]))
    ann_model.load_state_dict(checkpoint["state_dict"])

    # ann to snn
    snn_path = os.path.join(snn_args.train_dir, "snn.pth")
    if os.path.exists(snn_path):
        snn_state_dict = torch.load(snn_path)
        snn = create_model_snn(model=ann_model, batch_size=snn_args.batch_size, input_shape=in_shapes)
        snn.load_state_dict(snn_state_dict["state_dict"])
        logger.info("SNN model already exists")
    else:
        snn = convert(args, snn_args, ann_model, train_dataloader, in_shapes, device, logger)
        logger.info("[SNN]: {}".format(str(snn)))
    snn.to(device)
    snn.eval()

    # simulate
    # simulate(snn, train_dataloader, snn_args.timestep, num_classes, snn_args.train_dir, device)

    # hyperparameter search
    hyperparameter_search(args, snn_args, num_classes)

    # linear approximation
    approximate(snn_args)

def simulate(snn, train_dataloader, sim_time, num_classes, save_dir, device):
    tot = 0
    for i, data in enumerate(train_dataloader):
        if tot >= 10000:
            break

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        snn.reset()
            
        _out_spikes = simulate_iter(snn, images, sim_time, num_classes, device)

        np.savez_compressed(os.path.join(save_dir, "train", "labels_{}.npz".format(i)), l=labels.cpu().detach().numpy().copy())
        for index, out in _out_spikes.items():
            np.savez_compressed(os.path.join(save_dir, "train", "{}_output_{}.npz".format(index, i)), o=out.cpu().detach().numpy().copy())

        tot += labels.size(0)

@torch.no_grad()
def simulate_iter(snn, images, sim_time, num_classes, device):
    batch_size = images.size(0)

    out_spikes = {}
    for index in snn.classifiers.keys():
        out_spikes[int(index)] = torch.zeros(sim_time, batch_size, num_classes, device=device)

    for t in trange(sim_time):
        outputs = snn(images)

        for index, out in outputs.items():
            out_spikes[int(index)][t,:,:] = out

    return out_spikes
    

def hyperparameter_search(args, snn_args, num_classes):
    files_in_train_dir = os.listdir(os.path.join(snn_args.train_dir, "train"))
    labels_file_path = [f for f in files_in_train_dir if "labels_" in f]
    exp_num = len(labels_file_path)

    alphas = np.zeros((snn_args.timestep))
    betas = np.zeros((snn_args.timestep, 101))

    for i, labels_file in enumerate(labels_file_path):
        label_path_index = labels_file.split("_")[1].split(".")[0]
        final_output_path = os.path.join(snn_args.train_dir, "train", "-1_output_{}.npz".format(label_path_index))
        mid_output_path = os.path.join(snn_args.train_dir, "train", "{}_output_{}.npz".format(args.ic_index, label_path_index))

        labels = np.load(os.path.join(snn_args.train_dir, "train", labels_file))["l"]
        labels = labels[np.newaxis,:,np.newaxis]
        final_output = np.load(final_output_path)["o"]
        final_count = np.cumsum(final_output, axis=0)
        # print(final_output.shape, final_count.shape)
        # print(final_output[:20,0,0])
        # print(final_count[:20,0,0])
        mid_output = np.load(mid_output_path)["o"]
        mid_count = np.cumsum(mid_output, axis=0)

        # hyperparameter search
        time_batch = 50
        tmp_alphas = np.arange(0, 101, 1, dtype="float64")
        prior_alphas = np.tile(tmp_alphas, (time_batch, snn_args.batch_size, num_classes, 1))
        for t in range(int(snn_args.timestep//time_batch)):
            if snn_args.hps == "grid":
                N = np.arange(t*time_batch+1,(t+1)*time_batch+1, 1)
                N = N[:,np.newaxis,np.newaxis]
                M = mid_count[t*time_batch:(t+1)*time_batch,:,:]
                E = (final_count[t*time_batch:(t+1)*time_batch,:,:] / N) + 1e-16
                N = N[:,:,:,np.newaxis]
                M = M[:,:,:,np.newaxis]
                E = E[:,:,:,np.newaxis]

                best_acc = 0
                best_alpha = 1

                prior_betas = prior_alphas*(1/E-1)
                posterior_alphas = prior_alphas + M
                posterior_betas = prior_betas + (N - M)

                fr = posterior_alphas / (posterior_alphas + posterior_betas)
                pred = np.argmax(fr, axis=2)
                accs = np.sum(pred == labels, axis=1)
                best_alpha_index = np.argmax(accs, axis=-1)
                best_alpha = tmp_alphas[best_alpha_index]
            elif snn_args.hps == "emp":
                N = np.arange(t*time_batch+1,(t+1)*time_batch+1, 1)
                N = N[:,np.newaxis,np.newaxis]
                M = final_count[t*time_batch:(t+1)*time_batch,:,:]
                E = (mid_count[t*time_batch:(t+1)*time_batch,:,:] / N) + 1e-16
                N = N[:,:,:,np.newaxis]
                M = M[:,:,:,np.newaxis]
                E = E[:,:,:,np.newaxis]

                beta_1 = Beta(M+prior_alphas+1, N-M+prior_alphas*(1/E-1))
                beta_2 = Beta(prior_alphas+1, prior_alphas*(1/E-1))
                beta = beta_1/(beta_2+1e-9)
                best_alpha_index = np.argmax(beta, axis=-1)
                best_alpha = np.mean(tmp_alphas[best_alpha_index], axis=(1,2))
                betas[t*time_batch:(t+1)*time_batch,:] += np.mean(beta, axis=(1,2)) / exp_num
            else:
                raise NotImplementedError("hyperparameter search mode is not implemented")
            alphas[t*time_batch:(t+1)*time_batch] += best_alpha / exp_num  # mean

    simtime = np.arange(snn_args.timestep)
    ys = alphas
    np.savez(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(snn_args.hps)), alpha=ys)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(simtime, ys, c="r", s=5)
    fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_{}.svg".format(snn_args.hps)))


def approximate(snn_args):
    xs = np.arange(snn_args.timestep)
    xs_unit = xs / snn_args.timestep
    ys = np.load(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(snn_args.hps)))["alpha"]

    fig = plt.figure(figsize=(4.8, 3.75))
    # fig = plt.figure(figsize=(3.2, 2.5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c="r", s=5, label=r"$\alpha$")

    interpolate_index = np.concatenate([np.arange(0, 500, 10), np.arange(500, 3000, 100)])
    interpolate_index = np.append(interpolate_index, 2999)
            
    interpolate_f = interpolate.interp1d(interpolate_index, ys[interpolate_index], kind='linear')
    interpolated = interpolate_f(xs)
    np.savez(os.path.join(snn_args.train_dir, "division_linear_alpha_{}.npz".format(snn_args.hps)), y=interpolated)

    ax1.plot(xs, interpolated, c="b", label="approximation curve")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel(r"$\alpha$")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_with_curve_{}.svg".format(snn_args.hps)))

if __name__ == "__main__":
    main()
