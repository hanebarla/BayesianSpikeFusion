import pickle
import csv
import os
import shutil
import datetime
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import dataclasses

def save_checkpoint(state, is_best=False, filename='checkpoint.pth', bestname='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def plot_train_curv(losses, accs, ic_index, savedir):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    epochs = np.arange(len(losses[0]))
    ax1.plot(epochs, losses[0], label='train')
    ax1.plot(epochs, losses[1], label='val')
    if ic_index != -1:
        ax1.plot(epochs, losses[1][-1], label='val {}'.format(-1))
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, accs[0], label='train')
    ax2.plot(epochs, accs[1][ic_index], label='val {}'.format(ic_index))
    if ic_index != -1:
        ax1.plot(epochs, losses[1][-1], label='val {}'.format(-1))
    ax2.set_ylabel('Accuracy')

    ax2.set_xlabel('epoch')
    ax2.legend()

    figname = os.path.join(savedir, "loss_acc_curv.png")
    fig.savefig(figname, dpi=300)

def get_time(start_time, now_time):
    elapsed_time = now_time - start_time
    elapsed_hour = int(elapsed_time // 3600)
    elapsed_minute = int((elapsed_time % 3600) // 60)
    elapsed_second = int(elapsed_time % 3600 % 60)

    return str(elapsed_hour).zfill(2) + ":" + str(elapsed_minute).zfill(2) + ":" + str(elapsed_second).zfill(2)

def plot_activations(activations, dir, neuron_type, prefix=None, time=None):
    for k, v in activations.items():
        if isinstance(v, dict):
            plot_activations(v, dir, neuron_type, prefix=k, time=time)
        elif len(v.shape) == 4:
            for i in range(v.shape[1]):  # heads
                fig, ax = plt.subplots()
                divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
                cax = divider.append_axes('right', '5%', pad='3%')

                im = ax.imshow(v[0, i, ...], cmap='jet', interpolation='none', aspect="auto")

                fig.colorbar(im, cax=cax)
                fig.tight_layout()

                filename = neuron_type + ("_{}_".format(prefix) if prefix is not None else "_")
                filename += "{}_{}".format(k, i) + ("_{}".format(time) if time is not None else "")
                # print(filename)

                savename = os.path.join(dir, filename)
                fig.savefig(savename, dpi=300)
                plt.close()
        elif len(v.shape) == 3:
            fig, ax = plt.subplots()
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', '5%', pad='3%')

            im = ax.imshow(v[0, ...], cmap='jet', interpolation='none', aspect="auto")

            fig.colorbar(im, cax=cax)
            fig.tight_layout()

            filename = neuron_type + ("_{}_".format(prefix) if prefix is not None else "_")
            filename += k + ("_{}".format(time) if time is not None else "")
            # print(filename)

            savename = os.path.join(dir, filename)
            fig.savefig(savename, dpi=300)
            plt.close()
        elif len(v.shape) == 2:
            fig, ax = plt.subplots()

            x = np.array(range(v.shape[-1]))
            ax.bar(x, v[0, ...])

            fig.tight_layout()

            filename = neuron_type + ("_{}_".format(prefix) if prefix is not None else "_")
            filename += k + ("_{}".format(time) if time is not None else "")
            # print(filename)

            savename = os.path.join(dir, filename)
            fig.savefig(savename, dpi=300)
            plt.close()
