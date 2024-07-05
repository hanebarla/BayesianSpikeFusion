##############################################################################################################
# @article{moitra2022spikesim,
#   title={SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks},
#   author={Moitra, Abhishek and Bhattacharjee, Abhiroop and Kuang, Runcong and Krishnan, Gokul and Cao, Yu and Panda, Priyadarshini},
#   journal={arXiv preprint arXiv:2210.12899},
#   year={2022}
# }
# github url: https://github.com/Intelligent-Computing-Lab-Yale/SpikeSim/tree/main, 2024/01/28
##############################################################################################################

import math
import json

import torch
import numpy as np

from model_ann import model_factory
from model_snn import SpikingSDN, SpikingConv, SpikingDense, SpikingResBlock, SpikingAvgPool
from argument import get_args

def spikesim_energy(model, data_shape, timestep):

    ch = data_shape[0]
    dim = (data_shape[1], data_shape[2])
    in_ch_list = []
    out_ch_list = []
    in_dim_list = []
    out_dim_list = []
    kernel_list = []

    print("input", ch, dim)
    for module in model.feature:
        if isinstance(module, SpikingConv):
            in_ch_list.append(ch)
            in_dim_list.append(dim)

            out_ch, in_ch, k_h, k_w = module.weight.size()
            ch = out_ch
            dim = (int((dim[0] + 2*module.padding[0] - k_h)/module.stride[0]) + 1, 
                    int((dim[1] + 2*module.padding[1] - k_w)/module.stride[1]) + 1)
            
            out_ch_list.append(ch)
            out_dim_list.append(dim)
            kernel_list.append((k_h, k_w))
            # print(module, ch, dim)
            print("SpikingConv", ch, dim)
        elif isinstance(module, SpikingResBlock):
            ### conv_skip
            in_ch_list.append(ch)
            in_dim_list.append(dim)

            out_ch, in_ch, k_h, k_w = module.conv_skip.weight.size()
            skip_dim = (int((dim[0] + 2*module.conv_skip.padding[0] - k_h)/module.conv_skip.stride[0]) + 1,
                        int((dim[1] + 2*module.conv_skip.padding[1] - k_w)/module.conv_skip.stride[1]) + 1)

            out_ch_list.append(out_ch)
            out_dim_list.append(skip_dim)
            kernel_list.append((k_h, k_w))
            print("Skip Conv", out_ch, skip_dim)

            ### conv1
            in_ch_list.append(ch)
            in_dim_list.append(dim)

            out_ch, in_ch, k_h, k_w = module.conv1_weight.size()
            ch = out_ch
            dim = (int((dim[0] + 2*1 - k_h)/module.stride) + 1,
                   int((dim[1] + 2*1 - k_w)/module.stride) + 1)
            
            out_ch_list.append(ch)
            out_dim_list.append(dim)
            kernel_list.append((k_h, k_w))
            print("Conv1", ch, dim)

            ### conv2
            in_ch_list.append(ch)
            in_dim_list.append(dim)

            out_ch, in_ch, k_h, k_w = module.conv2.weight.size()
            ch = out_ch
            dim = (int((dim[0] + 2*module.conv2.padding[0] - k_h)/module.conv2.stride[0]) + 1,
                   int((dim[1] + 2*module.conv2.padding[1] - k_w)/module.conv2.stride[1]) + 1)
            
            out_ch_list.append(ch)
            out_dim_list.append(dim)
            kernel_list.append((k_h, k_w))
            print("Conv2", ch, dim)

    for _, modules in model.classifiers.items():
        for module in modules:
            if isinstance(module, SpikingAvgPool):
                in_ch_list.append(ch)
                in_dim_list.append(dim)

                out_ch_list.append(ch)
                out_dim_list.append((1,1))
                kernel_list.append((module.kernel_size, module.kernel_size))
                print("AvgPool", ch, dim)
            elif isinstance(module, SpikingDense):
                out_ch, in_ch = module.weight.size()
                
                parallel_size = 4
                in_ch_list.append(np.ceil(in_ch/parallel_size))
                in_dim_list.append((parallel_size, 1))

                out_ch_list.append(out_ch)
                out_dim_list.append((1,1))
                kernel_list.append((parallel_size,1))

                print("FC", out_ch, (1,1))

    print("in ch list: ", in_ch_list)
    print("in dim list: ", in_dim_list)
    print("out ch list: ", out_ch_list)
    print("out dim list: ", out_dim_list)

    with open('spikesim_config.json') as f:
        configs = json.load(f)

    tot_energy, energy_layer_wise = compute_energy(in_ch_list, out_ch_list, out_dim_list, kernel_list, timestep, configs)

    return tot_energy, energy_layer_wise


# count number of tiles and PEs
def pe_tile_count(in_ch_list, out_ch_list, configs):
    num_layer = len(out_ch_list)
    pe_list = []

    for i in range(num_layer):
        num_pe = np.ceil(in_ch_list[i] / configs["xbar_size"]) * np.ceil(out_ch_list[i] / configs["xbar_size"])
        pe_list.append(num_pe)

    if pe_list == sorted(pe_list):
        print("Layers in order")
    else:
        print("Check layer order")
        return

    num_pe = sum(pe_list)
    print(f'No. of PEs {num_pe}')

    num_tiles = math.ceil(num_pe / configs["pe_per_tile"])
    print(f'No. of Tiles {num_tiles}')

    return num_tiles


# All energies in pJ
def compute_energy(in_ch_list, out_ch_list, out_dim_list, kernel_list, time_steps, configs):
    assert len(in_ch_list) == len(out_ch_list) == len(kernel_list) == len(out_dim_list), "Check layer lengths"
    xbar_size = configs["xbar_size"]

    xbar_ar = configs["energy"]["xbar_ar"]
    Tile_buff = configs["energy"]["tile_buff"]
    Temp_buff = configs["energy"]["temp_buff"]
    Sub = configs["energy"]["sub"]
    ADC = configs["energy"]["adc"]

    Htree = configs["energy"]["htree"] * 8  # 4.912*4 3.11E+6/30.*0.25
    # Include PE dependent HTree
    MUX = configs["energy"]["mux"]
    mem_fetch = configs["energy"]["mem_fetch"]
    neuron = configs["energy"]["neuron"] * 4.0

    PE_cycle_energys = []
    for k in kernel_list:
        PE_ar = k[0] * k[1] * xbar_ar + (xbar_size/ 8) * (ADC + MUX)
        PE_cycle_energy = Htree + mem_fetch + neuron + xbar_size / 8 * PE_ar \
                            + (xbar_size / 8) * 16 * Sub + (xbar_size / 8) * Temp_buff + Tile_buff
        PE_cycle_energys.append(PE_cycle_energy)
        print('PE_cycle_energy {:.3e} pJ'.format(PE_cycle_energy))

    energy_layerwise = []
    tot_energy = 0
    tot_pe_cycle = 0
    for i, (in_ch, out_ch, PE_cycle_energy, out_dim) in enumerate(zip(in_ch_list, out_ch_list, PE_cycle_energys, out_dim_list)):
        Total_PE_cycle = np.ceil(out_ch / xbar_size) * np.ceil(in_ch / xbar_size) * (out_dim[0] * out_dim[1])
        print('Total_PE_cycle {:.3e}'.format(Total_PE_cycle))
        tot_energy += Total_PE_cycle * PE_cycle_energy * time_steps
        tot_pe_cycle += Total_PE_cycle
        energy_layerwise.append(Total_PE_cycle * PE_cycle_energy * time_steps)

    print('total_energy {:.3e} pJ'.format(tot_energy))
    return tot_energy, energy_layerwise

def main():
    args = get_args()

    if args.dataset == "mnist":
        data_shape = (1, 28, 28)
        num_classes = 10
    elif args.dataset == "cifar10":
        data_shape = (3, 32, 32)
        num_classes = 10
    elif args.dataset == "cifar100":
        data_shape = (3, 32, 32)
        num_classes = 100
    model = model_factory(args, num_classes, data_shape)
    snn = SpikingSDN(model, 128, data_shape)

    spikesim_energy(snn, data_shape, 1)

    # n_tiles = pe_tile_count(in_ch_list, out_ch_list, out_dim_list, kernel_size, xbar_size, pe_per_tile)
    # compute_energy(in_ch_list, in_dim_list, out_ch_list, out_dim_list, xbar_size, kernel_size, n_tiles, 'rram', time_steps)

if __name__ == "__main__":
    main()
