import os, argparse
import csv

from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np

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

    subdir = "snn{}".format("_{}".format(snn_args.init_mem) if snn_args.init_mem != 0.0 else "")
    if "dvs" in args.dataset:
        if snn_args.dvs_parallel:
            subdir += "_parallel"
        else:
            subdir += "_{}".format("sequential" if snn_args.sequential else "nonsequential")
    save_dir = os.path.join(snn_args.train_dir, subdir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create logger
    logger = create_logger(snn_args.train_dir, conditions, 'simulate.txt')
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
    # state_dict = checkpoint["state_dict"]
    # if "module" == list(state_dict.keys())[0][:6]:
    #     state_dict = fix_model_state_dict(state_dict)
    # ann_model.load_state_dict(state_dict)

    # ann to snn
    snn_path = os.path.join(snn_args.train_dir, "snn_{}.pth".format(snn_args.percentile))
    if os.path.exists(snn_path):
        snn_state_dict = torch.load(snn_path)
        snn = create_model_snn(model=ann_model, batch_size=snn_args.batch_size, input_shape=in_shapes, init_mem=snn_args.init_mem)
        snn.load_state_dict(snn_state_dict["state_dict"])
        logger.info("SNN model already exists")
    else:
        ann_model.to(device)
        snn = convert(args, snn_args, ann_model, train_dataloader, in_shapes, device, logger)
        logger.info("[SNN]: {}".format(str(snn)))
    snn.to(device)
    snn.eval()

    # simulate
    simulate(args, snn_args, snn, test_dataloader, snn_args.timestep, num_classes, save_dir, device, logger)

@torch.no_grad()
def simulate(args, snn_args, snn, test_dataloader, sim_time, num_classes, save_dir, device, logger):
    logger.info("Data num: {}".format(len(test_dataloader.dataset)))
    for i, data in enumerate(test_dataloader):
        logger.info("Progress: {}/{}".format(i+1, len(test_dataloader)))
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        snn.reset(batch_size=images.size(0))
        snn.to(device)
        
        if "dvs" in args.dataset:
            _out_spikes = simulate_iter_dvs(snn, images, sim_time, num_classes, device, snn_args.sequential, snn_args.dvs_parallel)
        else:
            _out_spikes = simulate_iter(snn, images, sim_time, num_classes, device)
        
        save_dict = {
            "labels": labels.cpu().detach().numpy().copy()
        }

        # np.savez_compressed(os.path.join(save_dir, "original", "labels_{}.npz".format(i)), l=labels.cpu().detach().numpy().copy())
        for index, out in _out_spikes.items():
            if index == -1:
                save_dict["final"] = out.cpu().detach().numpy().copy()
            else:
                save_dict["mid"] = out.cpu().detach().numpy().copy()
            # np.savez_compressed(os.path.join(save_dir, "original", "{}_output_{}.npz".format(index, i)), o=out.cpu().detach().numpy().copy())
        np.savez_compressed(os.path.join(save_dir, "output_{}.npz".format(i)), **save_dict)

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
                    out_spikes[int(index)][img_t,:,:] += out / T

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
  
if   __name__ == "__main__":
    main()
