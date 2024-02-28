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

def main():
    snn_args = get_snn_args()
    args = load_args(os.path.join(snn_args.train_dir, "command.json"))
    args.snn = True
    args.batch_size = snn_args.batch_size

    conditions, _ = get_save_snn_dir(args)

    if not os.path.exists(os.path.join(snn_args.train_dir, "snn")):
        os.makedirs(os.path.join(snn_args.train_dir, "snn"))

    # create logger
    logger = create_logger(snn_args.train_dir, conditions, 'simulate.txt'.format(snn_args.hps))
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
    simulate(snn, train_dataloader, snn_args.timestep, num_classes, snn_args.train_dir, device)
    
def simulate(snn, train_dataloader, sim_time, num_classes, save_dir, device):
    for i, data in enumerate(train_dataloader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        snn.reset()
        
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
        np.savez_compressed(os.path.join(save_dir, "snn", "output_{}.npz".format(i)), **save_dict)

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

"""
def simulate_snn(ic_index, work_dir, dataset_dir, sim_time):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
  
    print()
    print(" --- start snn simulation --- ")
  
    batch_size = 500
    train_dataloader, val_dataloader, test_dataloader, num_classes, input_shape = dataloader_factory(args.dataset, dataset_dir, batch_size, normalize=False, augument=True)
  
    ann_model = model_factory(model_kind=args.model, ic_index=ic_index, activation="relu", num_classes=num_classes, input_channel=input_shape[0])
    spiking_model = create_model_snn(model=ann_model, batch_size=batch_size, input_shape=input_shape)

    ann_model.load_state_dict(torch.load(os.path.join(work_dir, "ann.pth")))
    spiking_model.load_state_dict(torch.load(os.path.join(work_dir, "snn.pth")))
    
    tau = 0.5
  
    ann_model.to(device)
    ann_model.eval()
    spiking_model.to(device)
    spiking_model.eval()

    if ic_index != -1:
        # data = np.load(os.path.join(work_dir, "coe.npz"))
        # dim, coe = data["dim"], data["coe"]
        # data = np.load(os.path.join(work_dir, "gauss_approximate.npz"))
        # data = np.load(os.path.join(work_dir, "gaussian_process_approximate.npz"))
        data = np.load(os.path.join(work_dir, "division_linear_alpha_{}.npz".format(args.alpha_mode)))
        alphas = data["y"]
        # data = np.load(os.path.join(work_dir, "alpha_output.npz"))
        # alphas = data["alpha"]
        # alphas = np.full(sim_time, 50.0)
  
    with torch.no_grad():
        spikecount = torch.zeros(sim_time, device=device)
        spikecount_mid = torch.zeros(sim_time, device=device)
        spikecount_final = torch.zeros(sim_time, device=device)
        correct = torch.zeros(sim_time, device=device)
        correct_fusion = torch.zeros(sim_time, device=device)
        correct_mid = torch.zeros(sim_time, device=device)
        mse = torch.zeros(sim_time, device=device)
        mse_fusion = torch.zeros(sim_time, device=device)
        mse_mid = torch.zeros(sim_time, device=device)
        tot = 0
  
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            one_hot_label = F.one_hot(labels, num_classes=num_classes)
            spiking_model.reset()

            ann_pred = ann_model(images)
            ann_pred[-1] = F.softmax(ann_pred[-1], 1)
  
            tot += labels.size(0)
            print(tot)

            count = {}
            fire_burnin = {}
            outputs_cnt = {}
            for index in spiking_model.classifiers.keys():
                count[int(index)] = torch.zeros((batch_size, num_classes), device=device)
                fire_burnin[int(index)] = torch.zeros((BURNIN, batch_size, num_classes), device=device)
                outputs_cnt[int(index)] = torch.zeros((sim_time, batch_size, num_classes), device=device)
  
            for t in trange(sim_time):
                outputs = spiking_model(images)
                tmp1, tmp2, tmp3 = spiking_model.get_count(ic_index)
                # if t+1 < BURNIN:
                #   continue
                spikecount[t] += tmp1
                spikecount_mid[t] += tmp2
                spikecount_final[t] += tmp3
  
                for index, out in outputs.items():
                    # count[index] += out
                    # print(fire_burnin[index][1:,:,:].size(), torch.unsqueeze(out, 0).size())
                    outputs_cnt[index][t,:,:] = out
                    fire_burnin[index] = torch.cat((fire_burnin[index][1:,:,:], torch.unsqueeze(out,0)))
                    count[index] = torch.sum(fire_burnin[index], dim=0)
  
                N = (t+1)
                M = count[-1]
                E = (count[ic_index] / N) + 1e-16
  
                if ic_index != -1:
                    # prior_alpha = quation_LSM(coe, [t/3000.0])[0]
                    prior_alpha = alphas[t]
                    prior_beta = prior_alpha*(1/E - 1)
                    posterior_alpha = prior_alpha + M
                    posterior_beta = prior_beta + (N - M)
                    fr = posterior_alpha / (posterior_alpha + posterior_beta)
                    variance = (posterior_alpha*posterior_beta) / (torch.pow(posterior_alpha*posterior_beta, 2) * (posterior_alpha+posterior_beta+1))  # var of Beta Distoribution
                    std = torch.sqrt(variance)
                else:
                    fr = count[-1] / N
                    std = torch.std(fire_burnin[-1] - fr, (1,2))

                pred_fusion = torch.max(fr, 1)[1]
                correct_fusion[t] += torch.sum(pred_fusion == labels)
                # mse_fusion[t] += F.mse_loss(one_hot_label, fr) * batch_size
                mse_fusion[t] += F.mse_loss(ann_pred[-1], fr) * batch_size
    
                pred = torch.max(count[-1], 1)[1]
                correct[t] += torch.sum(pred == labels)
                # mse[t] += F.mse_loss(one_hot_label, count[-1]/N) * batch_size
                mse[t] += F.mse_loss(ann_pred[-1], count[-1]/N) * batch_size

                pred_mid = torch.max(count[ic_index], 1)[1]
                correct_mid[t] += torch.sum(pred_mid == labels)
                # mse_mid[t] += F.mse_loss(one_hot_label, count[ic_index]/N) * batch_size
                mse_mid[t] += F.mse_loss(ann_pred[-1], count[ic_index]/N) * batch_size
                
            for index, cnt in outputs_cnt.items():
                np.savez_compressed(os.path.join(work_dir,"{}_{}_simulate_output_{}.npz".format(args.alpha_mode, index, i)), o=cnt.cpu().detach().numpy().copy(), l=labels.cpu().detach().numpy().copy())

            # if tot >= 2000:
            #   break
  
    print("tot:", tot)
    spikecount = spikecount.cpu().detach().numpy()
    spikecount_mid = spikecount_mid.cpu().detach().numpy()
    spikecount_final = spikecount_final.cpu().detach().numpy()
    correct = correct.detach().cpu().numpy()
    correct_fusion = correct_fusion.detach().cpu().numpy()
    correct_mid = correct_mid.detach().cpu().numpy()
    mse = mse.detach().cpu().numpy()
    mse_fusion = mse_fusion.detach().cpu().numpy()
    mse_mid = mse_mid.detach().cpu().numpy()
    acc = correct / tot
    acc_fusion = correct_fusion / tot
    acc_mid = correct_mid / tot
    mse = mse / tot
    mse_fusion = mse_fusion / tot
    mse_mid = mse_mid / tot
    print("acc:", acc)
    print("acc_mid:", acc_mid)
    print("acc_fusion:", acc_fusion)
    print("mse:", mse)
    print("mse_mid:", mse_mid)
    print("mse_fusion:", mse_fusion)
  
    np.savez(os.path.join(work_dir, "snn_result_{}.npz".format(args.alpha_mode)),
             acc=acc,
             acc_f=acc_fusion, 
             acc_m=acc_mid, 
             spikecount=spikecount_final, 
             spikecount_f=spikecount, 
             spikecount_m=spikecount_mid,
             mse=mse,
             mse_f=mse_fusion,
             mse_m=mse_mid)
"""
  
if   __name__ == "__main__":
    main()
