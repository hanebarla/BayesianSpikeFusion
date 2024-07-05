import os
import csv
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from argument import load_args
from model_ann import model_factory
from model_snn import SpikingSDN
from spikesim_ene import spikesim_energy

# ggplot style
plt.style.use('ggplot')

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('exps', nargs='*', type=str)
    args = parser.parse_args()

    return args

def plot_differ_mp(args):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    root_dir = os.path.dirname(args.exps[0] if args.exps[0][-1] != "/" else args.exps[0][:-1])

    # energy per one timestep
    ann_args = load_args(os.path.join(root_dir, "command.json"))
    if ann_args.dataset == "mnist":
        data_shape = (1, 28, 28)
        num_classes = 10
    elif ann_args.dataset == "cifar10":
        data_shape = (3, 32, 32)
        num_classes = 10
    elif ann_args.dataset == "cifar100":
        data_shape = (3, 32, 32)
        num_classes = 100
    elif ann_args.dataset == "tinyimagenet":
        data_shape = (3, 64, 64)
        num_classes = 200
    elif ann_args.dataset == "dvsgesture":
        data_shape = (2, 128, 128)
        num_classes = 11
    else:
        raise ValueError("Invalid dataset name")
    model = model_factory(ann_args, num_classes, data_shape)
    snn = SpikingSDN(model, 128, data_shape)
    energy_per_time, layer_energy_per_time = spikesim_energy(snn, data_shape, 1)
    BSF_energy_per_time = energy_per_time
    FP_energy_per_time = sum(layer_energy_per_time[:-2]+[layer_energy_per_time[-1]])
    # pj to J
    BSF_energy_per_time *= 1e-12
    FP_energy_per_time *= 1e-12
    print("BSF_energy: {:.4e}, FP_energy: {:.4e}".format(BSF_energy_per_time, FP_energy_per_time))

    grid_alphas = np.load(os.path.join(root_dir, "division_linear_alpha_grid_sequential.npz"))["y"]  # timestep,
    emp_alphas = np.load(os.path.join(root_dir, "division_linear_alpha_emp_sequential.npz"))["y"]  # timestep,
    # grid_alphas = np.load(os.path.join(root_dir, "alpha_output_grid_parallel.npz"))["alpha"]  # timestep,
    # emp_alphas = np.load(os.path.join(root_dir, "alpha_output_emp_parallel.npz"))["alpha"]  # timestep,

    ann_args = load_args(os.path.join(root_dir, "command.json"))
    with open(os.path.join(root_dir, "ann_acc.csv"), "r") as f:
        reader = csv.reader(f)
        ann_acc = list(reader)
        print(ann_acc)
        fp_acc = float(ann_acc[-1][-1])
    print("fp_acc: ", fp_acc)

    for exp in args.exps:
        ene_acc_file = os.path.join(exp, "ene_acc.npz")
        if os.path.exists(ene_acc_file):
            data = np.load(ene_acc_file)
            grid_acc = data["grid_acc"]
            emp_acc = data["emp_acc"]
            # fp_energies = data["fp_energies"]
            fp_energies = np.arange(1, grid_acc.shape[0]+1) * FP_energy_per_time
            # bsf_energies = data["bsf_energies"]
            bsf_energies = np.arange(1, grid_acc.shape[0]+1) * BSF_energy_per_time
            final_acc = data["final_acc"]
        else:
            files = os.listdir(exp)
            files = [os.path.join(exp, file) for file in files if file.endswith(".npz")]
            files = [file for file in files if "ene_acc" not in file]
            grid_acc, emp_acc, final_acc, mid_acc, fp_energies, bsf_energies = get_ene_acc(files, grid_alphas, emp_alphas, FP_energy_per_time, BSF_energy_per_time)
            np.savez_compressed(os.path.join(exp, "ene_acc.npz"), grid_acc=grid_acc, emp_acc=emp_acc, final_acc=final_acc, mid_acc=mid_acc, fp_energies=fp_energies, bsf_energies=bsf_energies)

        ax.plot(bsf_energies, grid_acc, label="grid")
        ax.plot(bsf_energies, emp_acc, label="emp")
        ax.plot(fp_energies, final_acc, label="final")

    percentage = 0.97
    target_acc = fp_acc*percentage
    print(fp_acc, target_acc)

    fp_exceed_indexes = final_acc >= target_acc
    fp_exceed_energies = fp_energies[fp_exceed_indexes][0]
    # print(fp_exceed_energies)
    bsf_exceed_indexes_gird = grid_acc >= target_acc
    bsf_exceed_energies_gird = bsf_energies[bsf_exceed_indexes_gird][0]
    # print(bsf_exceed_energies_gird)
    bsf_exceed_indexes_emp = emp_acc >= target_acc
    bsf_exceed_energies_emp = bsf_energies[bsf_exceed_indexes_emp][0]
    # print(bsf_exceed_energies_emp)

    print(final_acc[fp_exceed_indexes][0], grid_acc[fp_exceed_indexes][0], emp_acc[fp_exceed_indexes][0])
    print(fp_exceed_energies, bsf_exceed_energies_gird, bsf_exceed_energies_emp)
    fp_acc_target, fp_auc = calc_auc(fp_energies, final_acc, fp_exceed_energies)
    bsf_acc_grid, bsf_auc_grid = calc_auc(bsf_energies, grid_acc, fp_exceed_energies)
    bsf_acc_emp, bsf_auc_emp = calc_auc(bsf_energies, emp_acc, fp_exceed_energies)
    print("fp_acc_target: {:.4f}, fp_auc: {:.4e}, fp_ene: {:.4e}".format(fp_acc_target, fp_auc, fp_exceed_energies))
    print("bsf_acc_grid: {:.4f}, bsf_auc_grid: {:.4e}, bsf_ene_grid: {:.4e}".format(bsf_acc_grid, bsf_auc_grid, bsf_exceed_energies_gird))
    print("bsf_acc_emp: {:.4f}, bsf_auc_emp: {:.4e}, bsf_ene_emp: {:.4e}".format(bsf_acc_emp, bsf_auc_emp, bsf_exceed_energies_emp))
    print("AUC increse, Grid: {:.1f}, Emp: {:.1f}".format((bsf_auc_grid-fp_auc)/fp_auc*100, (bsf_auc_emp-fp_auc)/fp_auc*100))
    print("Energy decrease, Grid: {:.1f}, Emp: {:.1f}".format((fp_exceed_energies-bsf_exceed_energies_gird)/fp_exceed_energies*100, (fp_exceed_energies-bsf_exceed_energies_emp)/fp_exceed_energies*100))

    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("timestep")
    ax.set_ylabel("accuracy")

    if not os.path.exists("Plot"):
        os.makedirs("Plot")
    fig.savefig(os.path.join("Plot", "ene_acc.svg"))

def calc_auc(energies, accs, target_ene):
    dx = 1e-7
    x = energies[energies<=target_ene]
    y = accs[energies<=target_ene]
    auc = integrate.trapezoid(y, x, dx=dx)

    return y[-1], auc

def get_ene_acc(files, grid_alphas, emp_alphas, fp_energy_per_time, bsf_energy_per_time):
    tot = 0
    grid_acc = np.zeros(grid_alphas.shape[0])
    emp_acc = np.zeros(emp_alphas.shape[0])
    final_acc = np.zeros(grid_alphas.shape[0])
    mid_acc = np.zeros(emp_alphas.shape[0])
    timesteps = np.arange(1, grid_alphas.shape[0]+1)
    fp_energies = timesteps * fp_energy_per_time
    bsf_energies = timesteps * bsf_energy_per_time

    for i, file in enumerate(files):
        data = np.load(file)
        labels = data["labels"]
        final = data["final"]  # timestep, batch, class
        mid = data["mid"]
        # print(final.shape, mid.shape)

        final_cnt = final.cumsum(axis=0)
        mid_cnt = mid.cumsum(axis=0)
        final_fr = final_cnt / np.arange(1, final_cnt.shape[0]+1)[:,np.newaxis,np.newaxis]
        mid_fr = mid_cnt / np.arange(1, mid_cnt.shape[0]+1)[:,np.newaxis,np.newaxis]
        # print(final_cnt.shape, mid_cnt.shape)
        # print(final[:20,-1,-1])
        # print(final_cnt[:20,-1,-1])

        grid_fr, grid_std = calc_fr_std(grid_alphas, final_cnt, mid_cnt)
        emp_fr, emp_std = calc_fr_std(emp_alphas, final_cnt, mid_cnt)

        grid_acc += np.sum(np.argmax(grid_fr, axis=2) == labels, axis=1)
        emp_acc += np.sum(np.argmax(emp_fr, axis=2) == labels, axis=1)
        final_acc += np.sum(np.argmax(final_fr, axis=2) == labels, axis=1)
        mid_acc += np.sum(np.argmax(mid_fr, axis=2) == labels, axis=1)

        print(labels.shape[0], grid_acc[-1], emp_acc[-1], final_acc[-1], mid_acc[-1])
        tot += labels.shape[0]
        print("Progress: {}/{}, Tot: {}, grid_acc: {:.4f}, emp_acc: {:.4f}".format(i+1, len(files), tot, grid_acc[-1]/tot, emp_acc[-1]/tot))

    grid_acc /= tot
    emp_acc /= tot
    final_acc /= tot
    mid_acc /= tot

    return grid_acc, emp_acc, final_acc, mid_acc, fp_energies, bsf_energies
        
def calc_fr_std(alphas, final, mid):
    N = np.arange(1, final.shape[0]+1)[:,np.newaxis,np.newaxis]
    M = final
    E = (mid / N) + 1e-16

    alphas = alphas[:, np.newaxis, np.newaxis]
    
    prior_alphas = alphas
    prior_betas = prior_alphas*(1/E-1)
    posterior_alphas = prior_alphas + M
    posterior_betas = prior_betas + (N - M)
    fr = posterior_alphas / (posterior_alphas + posterior_betas)
    variance = (posterior_alphas * posterior_betas) / ((posterior_alphas + posterior_betas)**2 * (posterior_alphas + posterior_betas + 1))
    std = np.sqrt(variance)

    return fr, std

def main():
    args = create_args()
    
    plot_differ_mp(args)

if __name__ == "__main__":
    main()
