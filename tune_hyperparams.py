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
from utils import fix_model_state_dict

def main():
    snn_args = get_snn_args()
    args = load_args(os.path.join(snn_args.train_dir, "command.json"))
    args.snn = True
    args.batch_size = snn_args.batch_size

    conditions, _ = get_save_snn_dir(args)

    # create logger
    logger = create_logger(snn_args.train_dir, conditions, 'tune_log.txt')
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
    args.batch_norm = False
    ann_model = model_factory(args, num_classes, in_shapes)
    logger.info("[model]: {}".format(str(ann_model)))
    # checkpoint = torch.load(os.path.join(snn_args.train_dir, "checkpoint.pth"))
    # logger.info("ANN Accuracy: {}".format(checkpoint["acc"]))
    # if list(checkpoint["state_dict"].keys())[0].startswith("module."):
    #     ann_model.load_state_dict(fix_model_state_dict(checkpoint["state_dict"]))
    # else:
    #     ann_model.load_state_dict(checkpoint["state_dict"])

    # ann to snn
    snn_path = os.path.join(snn_args.train_dir, "snn_{}.pth".format(snn_args.percentile))
    # print(os.path.exists(snn_path))
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
    if "dvs" in args.dataset:
        post_fix = ""
        if snn_args.dvs_parallel:
            post_fix = "_parallel"
        elif snn_args.sequential:
            post_fix = "_sequential"
        else:
            post_fix = "_nonsequential"
        train_sim_dir = os.path.join(snn_args.train_dir, "train" + post_fix)
    elif snn_args.init_mem == 0.0:
        train_sim_dir = os.path.join(snn_args.train_dir, "train")
    else:
        train_sim_dir = os.path.join(snn_args.train_dir, "train_init_mem_{}".format(snn_args.init_mem))
    if not os.path.exists(train_sim_dir):
        os.makedirs(train_sim_dir)
    simulate(args, snn_args, snn, train_dataloader, snn_args.timestep, num_classes, train_sim_dir, device)

    # hyperparameter search
    logger.info("Hyperparameter search (Grid)")
    hyperparameter_search(args, snn_args, num_classes, "grid", train_sim_dir, logger)
    logger.info("Hyperparameter search (Empirical)")
    hyperparameter_search(args, snn_args, num_classes, "emp", train_sim_dir, logger)

    # linear approximation
    logger.info("Approximate Grid")
    approximate(args, snn_args, "grid")
    logger.info("Approximate Empirical")
    approximate(args, snn_args, "emp")
    logger.info("Done")

def simulate(args, snn_args, snn, train_dataloader, sim_time, num_classes, save_dir, device):
    tot = 0
    for i, data in enumerate(train_dataloader):
        print("iter: ", i)
        if tot >= 10000:
            break

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        snn.reset()
            
        if "dvs" in args.dataset:
            _out_spikes = simulate_iter_dvs(snn, images, sim_time, num_classes, device, snn_args.sequential, snn_args.dvs_parallel)
        else:
            _out_spikes = simulate_iter(snn, images, sim_time, num_classes, device)

        np.savez_compressed(os.path.join(save_dir, "labels_{}.npz".format(i)), l=labels.cpu().detach().numpy().copy())
        for index, out in _out_spikes.items():
            np.savez_compressed(os.path.join(save_dir, "{}_output_{}.npz".format(index, i)), o=out.cpu().detach().numpy().copy())

        tot += labels.size(0)

@torch.no_grad()
def simulate_iter(snn, images, sim_time, num_classes, device):
    batch_size = images.size(0)
    snn.reset(batch_size=batch_size)

    out_spikes = {}
    for index in snn.classifiers.keys():
        out_spikes[int(index)] = torch.zeros(sim_time, batch_size, num_classes, device=device)

    for t in trange(sim_time):
        outputs = snn(images)

        for index, out in outputs.items():
            out_spikes[int(index)][t,:,:] = out

    return out_spikes

@torch.no_grad()
def simulate_iter_dvs(snn, images, sim_time, num_classes, device, sequential=True, dvs_parallel=False):
    N, T, C, H, W = images.size()
    snn.reset(batch_size=N)
    snn.to(device)

    out_spikes = {}
    for index in snn.classifiers.keys():
        out_spikes[int(index)] = torch.zeros(sim_time, N, num_classes, device=device)

    if dvs_parallel:
        for img_t in trange(T):
            snn.reset(batch_size=N)
            snn.to(device)
            for t in range(sim_time):
                outputs = snn(images[:,img_t,:,:,:])

                for index, out in outputs.items():
                    out_spikes[int(index)][t,:,:] += out / T

        return out_spikes

    if sequential:
        for t in trange(sim_time//T):
            for img_t in range(T):
                outputs = snn(images[:,img_t,:,:,:])

                for index, out in outputs.items():
                    out_spikes[int(index)][t,:,:] = out
    else:
        for img_t in trange(T):
            for t in range(sim_time//T):
                outputs = snn(images[:,img_t,:,:,:])

                for index, out in outputs.items():
                    out_spikes[int(index)][t,:,:] = out

    return out_spikes
    

def hyperparameter_search(args, snn_args, num_classes, hps, saved_dir, logger):
    files_in_train_dir = os.listdir(os.path.join(saved_dir))
    labels_file_path = [f for f in files_in_train_dir if "labels_" in f]
    exp_num = len(labels_file_path)

    alphas = np.zeros((snn_args.timestep))
    betas = np.zeros((snn_args.timestep, 101))

    for i, labels_file in enumerate(labels_file_path):
        logger.info("Hyperparameter search: {}/{}".format(i+1, exp_num))
        label_path_index = labels_file.split("_")[1].split(".")[0]
        final_output_path = os.path.join(saved_dir, "-1_output_{}.npz".format(label_path_index))
        mid_output_path = os.path.join(saved_dir, "{}_output_{}.npz".format(args.ic_index, label_path_index))

        labels = np.load(os.path.join(saved_dir, labels_file))["l"]
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
        tmp_alphas = np.arange(0, 101, 1, dtype="float32")
        prior_alphas = np.tile(tmp_alphas, (time_batch, snn_args.batch_size, num_classes, 1))
        for t in trange(int(snn_args.timestep//time_batch)):
            if hps == "grid":
                N = np.arange(t*time_batch+1,(t+1)*time_batch+1, 1)
                N = N[:,np.newaxis,np.newaxis]
                M = final_count[t*time_batch:(t+1)*time_batch,:,:]
                E = (mid_count[t*time_batch:(t+1)*time_batch,:,:] / N) + 1e-16
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
            elif hps == "emp":
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

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(simtime, ys, c="r", s=5)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel(r"$\alpha$")

    if "dvs" in args.dataset:
        file_name = "alpha_output_{}".format(hps)
        if snn_args.dvs_parallel:
            file_name += "_parallel"
        elif snn_args.sequential:
            file_name += "_sequential"
        else:
            file_name += "_nonsequential"
        file_name += ".npz"
        np.savez(os.path.join(snn_args.train_dir, file_name), alpha=ys)
        fig.savefig(os.path.join(snn_args.train_dir, file_name.replace(".npz", ".svg")))
    elif snn_args.init_mem == 0.0:
        np.savez(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(hps)), alpha=ys)
        fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_{}.svg".format(hps)))
    else:
        np.savez(os.path.join(snn_args.train_dir, "alpha_output_{}_init_mem_{}.npz".format(hps, snn_args.init_mem), alpha=ys))
        fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_{}_init_mem_{}.svg".format(hps, snn_args.init_mem)))
        # np.savez(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(hps)), alpha=ys)
    # fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_{}.svg".format(hps)))


def approximate(args, snn_args, hps):
    xs = np.arange(snn_args.timestep)
    xs_unit = xs / snn_args.timestep
    if "dvs" in args.dataset:
        file_name = "alpha_output_{}".format(hps)
        if snn_args.dvs_parallel:
            file_name += "_parallel"
        elif snn_args.sequential:
            file_name += "_sequential"
        else:
            file_name += "_nonsequential"
        file_name += ".npz"
        ys = np.load(os.path.join(snn_args.train_dir, file_name))["alpha"]
    elif snn_args.init_mem == 0.0:
        ys = np.load(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(hps)))["alpha"]
    else:
        ys = np.load(os.path.join(snn_args.train_dir, "alpha_output_{}_init_mem_{}.npz".format(hps, snn_args.init_mem)))["alpha"]
    # ys = np.load(os.path.join(snn_args.train_dir, "alpha_output_{}.npz".format(snn_args.hps)))["alpha"]

    fig = plt.figure(figsize=(4.8, 3.75))
    # fig = plt.figure(figsize=(3.2, 2.5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c="r", s=5, label=r"$\alpha$")

    interpolate_index = np.concatenate([np.arange(0, 500, 10), np.arange(500, 3000, 100)])
    interpolate_index = np.append(interpolate_index, 2999)
            
    interpolate_f = interpolate.interp1d(interpolate_index, ys[interpolate_index], kind='linear')
    interpolated = interpolate_f(xs)

    if "dvs" in args.dataset:
        file_name = "division_linear_alpha_{}".format(hps)
        if snn_args.dvs_parallel:
            file_name += "_parallel"
        elif snn_args.sequential:
            file_name += "_sequential"
        else:
            file_name += "_nonsequential"
        file_name += ".npz"
        np.savez(os.path.join(snn_args.train_dir, file_name), y=interpolated)
    elif snn_args.init_mem == 0.0:
        np.savez(os.path.join(snn_args.train_dir, "division_linear_alpha_{}.npz".format(hps)), y=interpolated)
    else:
        np.savez(os.path.join(snn_args.train_dir, "division_linear_alpha_{}_init_mem_{}.npz".format(hps, snn_args.init_mem)), y=interpolated)
    # np.savez(os.path.join(snn_args.train_dir, "division_linear_alpha_{}.npz".format(snn_args.hps)), y=interpolated)

    ax1.plot(xs, interpolated, c="b", label="approximation curve")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel(r"$\alpha$")
    ax1.legend()
    fig.tight_layout()
    if "dvs" in args.dataset:
        file_name = "time_alpha_with_curve_{}".format(hps)
        if snn_args.dvs_parallel:
            file_name += "_parallel"
        elif snn_args.sequential:
            file_name += "_sequential"
        else:
            file_name += "_nonsequential"
        file_name += ".svg"
        fig.savefig(os.path.join(snn_args.train_dir, file_name))
    if snn_args.init_mem == 0.0:
        fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_with_curve_{}.svg".format(hps)))
    else:
        fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_with_curve_{}_init_mem_{}.svg".format(hps, snn_args.init_mem)))
    # fig.savefig(os.path.join(snn_args.train_dir, "time_alpha_with_curve_{}.svg".format(snn_args.hps)))

if __name__ == "__main__":
    main()
