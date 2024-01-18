import os, argparse
import torch
import torch.nn.functional as F
import numpy as np
import csv

from dataset import dataset_factory
from model_ann import model_factory
from model_snn import create_model_snn
from lsm import quation_LSM
from tqdm import trange
from threshold import vth_func_factory


parser = argparse.ArgumentParser()
parser.add_argument('model', default='vgg11', type=str)
parser.add_argument('dataset', default='cifar10', type=str)
parser.add_argument('alpha_mode', type=str)
parser.add_argument('--vth', default="None", type=str)
parser.add_argument('--ic_index', type=int, default=-1)
parser.add_argument('--sim_time', type=int, default=3000)
parser.add_argument('--plot_alpha', type=int, default=1)
parser.add_argument('--encoder', default="direct")  # direct, bernoulli, dither, dither_bernoulli
args = parser.parse_args()


BURNIN = 500

def encoder(images, img_err, mode="bernoulli"):
    print(mode)
    if mode == "direct":
        return images, img_err
    elif mode == "bernoulli":
        return torch.bernoulli(images), img_err
    elif "dither" in mode:
        b, c, H, W = images.size()
        new_images = torch.zeros_like(images)
        err = torch.zeros((b, c), device=images.device)
        for h in range(H):
            for w in range(W):
                tmp_value = images[:,:,h,w] + img_err[:,:,h,w] + err
                if "bernoulli" in mode:
                    new_images[:,:,h,w] = torch.bernoulli(torch.clamp(tmp_value, 0.0, 1.0))
                else:
                    new_images[:,:,h,w] = tmp_value > 0.5
                err -= new_images[:,:,h,w]
                err += images[:,:,h,w]    
        return new_images, img_err
    else:
        raise NotImplementedError(mode)


def simulate_snn(ic_index, work_dir, dataset_dir, sim_time):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
  
    print()
    print(" --- start snn simulation --- ")
  
    batch_size = 500
    train_dataloader, test_dataloader, num_classes, input_shape = dataset_factory(args.dataset, dataset_dir, batch_size, normalize=False, augument=True)
  
    ann_model = model_factory(model_kind=args.model, ic_index=ic_index, activation="relu", num_classes=num_classes, input_channel=input_shape[0])
    spiking_model = create_model_snn(model=ann_model, batch_size=batch_size, input_shape=input_shape)

    ann_model.load_state_dict(torch.load(os.path.join(work_dir, "ann.pth")))
    spiking_model.load_state_dict(torch.load(os.path.join(work_dir, "snn.pth")))
    
    tau = 0.5
    vth_func = vth_func_factory(args.vth, tau, num_classes)
    
    if vth_func is not None:
        with open(os.path.join(work_dir, "vth.csv"), "w") as f:
            print("vth file init")
  
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
        print(vth_func)
  
        for i, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            one_hot_label = F.one_hot(labels, num_classes=num_classes)
            spiking_model.reset()

            ann_pred = ann_model(images)
            ann_pred[-1] = F.softmax(ann_pred[-1], 1)
            
            vth = 1.0
            spiking_model.set_vth(vth=vth)
            if str(vth_func) == "automaton":
                vth_func.state = 0
  
            tot += labels.size(0)
            print(tot)
  
            vth_list = [vth]
            count = {}
            fire_burnin = {}
            outputs_cnt = {}
            for index in spiking_model.classifiers.keys():
                count[int(index)] = torch.zeros((batch_size, num_classes), device=device)
                fire_burnin[int(index)] = torch.zeros((BURNIN, batch_size, num_classes), device=device)
                outputs_cnt[int(index)] = torch.zeros((sim_time, batch_size, num_classes), device=device)
  
            img_err = torch.zeros_like(images)
            for t in trange(sim_time):
                images_bin, img_err = encoder(images, img_err, mode=args.encoder)
                outputs = spiking_model(images_bin)
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
                
                if vth_func is not None:
                    # fired = torch.sum(torch.abs(outputs[index]), dim=1)
                    # watchSpikeCount = min((torch.mean(fired)/vth).item()/(t+1), 1)  # mean fire count
                    # vth = vth_func(watchSpikeCount, vth)
                    entropy_time = int(tau * 1000)
                    fire = torch.sum(fire_burnin[-1][-1-entropy_time:-1,:,:], dim=0)
                    fire_rate = F.softmax(fire, dim=-1)
                    entropy_per_node = -1 * fire_rate * torch.log2(fire_rate + 1e-5)
                    entropy = torch.sum(entropy_per_node, dim=-1)
                    entropy_mean = torch.mean(entropy)
                    vth = vth_func(vth=vth, entropy=entropy_mean)
                    spiking_model.set_vth(vth=vth)
                    vth_list.append(vth)
                
            for index, cnt in outputs_cnt.items():
                # np.savez_compressed(os.path.join(work_dir,"{}_simulate_output_{}{}.npz".format(index, i, "_" + str(vth_func) if vth_func is not None else "")), o=cnt.cpu().detach().numpy().copy(), l=labels.cpu().detach().numpy().copy())
                np.savez_compressed(os.path.join(work_dir,"{}_{}_simulate_output_{}{}.npz".format(args.alpha_mode, index, i, "_" + args.encoder)), o=cnt.cpu().detach().numpy().copy(), l=labels.cpu().detach().numpy().copy())
                
            if vth_func is not None:
                with open(os.path.join(work_dir, "vth.csv"), "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(vth_list)
  
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
  
    # np.savez(os.path.join(work_dir, "snn_result_{}{}.npz".format(args.alpha_mode, "_" + str(vth_func) if vth_func is not None else "")),
    #          acc=acc,
    #          acc_f=acc_fusion, 
    #          acc_m=acc_mid, 
    #          spikecount=spikecount_final, 
    #          spikecount_f=spikecount, 
    #          spikecount_m=spikecount_mid,
    #          mse=mse,
    #          mse_f=mse_fusion,
    #          mse_m=mse_mid)
    np.savez(os.path.join(work_dir, "snn_result_{}{}.npz".format(args.alpha_mode, "_" + args.encoder)),
             acc=acc,
             acc_f=acc_fusion, 
             acc_m=acc_mid, 
             spikecount=spikecount_final, 
             spikecount_f=spikecount, 
             spikecount_m=spikecount_mid,
             mse=mse,
             mse_f=mse_fusion,
             mse_m=mse_mid)
  
if   __name__ == "__main__":
    work_dir = os.path.join("/home/work/thabara/sdn", "models", "sdn-{}_{}".format(args.model, args.dataset), "{}".format(args.ic_index))
    # os.makedirs(work_dir, exist_ok=True)
  
    dataset_dir = os.path.join("/home/work/thabara/sdn", "datasets")
  
    simulate_snn(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir, sim_time=args.sim_time)
