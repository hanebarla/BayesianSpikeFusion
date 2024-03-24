import os
import json
import argparse
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()

    ########################################
    # Dataset setting
    ########################################
    g_dataset = parser.add_argument_group('Dataset')
    g_dataset.add_argument('dataset')
    g_dataset.add_argument('--data_path', default="/home/work/thabara/Datasets")
    g_dataset.add_argument('--batch_size', default=128, type=int)
    g_dataset.add_argument('--num_workers', default=2, type=int)
    g_dataset.add_argument('--prefetch', default=8, type=int)
    g_dataset.add_argument('--pin_memory', default=1, type=int)  # boolean
    g_dataset.add_argument('--split', default=0.8, type=float)
    g_dataset.add_argument('--mixup', default=0, type=float)  # 0.8
    g_dataset.add_argument('--snn', default=0, type=int)  # boolean

    ########################################
    # Model Setting
    ########################################
    g_model = parser.add_argument_group('Model')
    g_model.add_argument('model')
    g_model.add_argument('--ic_index', type=int, default=-1)
    g_model.add_argument('--activation', default='relu')
    g_model.add_argument('--dropout', default=0.0, type=float)
    g_model.add_argument('--batch_norm', default=1, type=int)  # boolean
    g_model.add_argument('--label_smoothing', default=0, type=float)
    g_model.add_argument('--pretrained', type=str, default="")
    g_model.add_argument('--qcfs', default=0, type=int)  # floor 

    ########################################
    # ANN Train Setting
    ########################################
    g_train = parser.add_argument_group('Train')
    g_train.add_argument('--epochs', default=200, type=int)  # 200
    ## optimizer
    g_train.add_argument('--optimizer', default='sgd')
    g_train.add_argument('--opt_wd', default=5e-4, type=float)
    g_train.add_argument('--opt_eps', default=1e-8, type=float)
    g_train.add_argument('--opt_betas', default=(0.9, 0.999), type=tuple)
    g_train.add_argument('--opt_moment', default=0.9, type=float)
    ## schduler
    g_train.add_argument('--lr_sch', default='multistep')
    g_train.add_argument('--lr', default=0.1, type=float)
    ## cosine sch
    g_train.add_argument('--lr_t_initial', default=100, type=int)
    g_train.add_argument('--lr_min', default=1e-5, type=float)
    g_train.add_argument('--warmup_t', default=3, type=int)
    g_train.add_argument('--warmup_lr_init', default=1e-5, type=float)
    ## multistep sch
    g_train.add_argument('--lr_steps', default=[60, 120, 160], nargs='*', type=int)  # 60, 120, 160
    g_train.add_argument('--gamma', default=0.2, type=float)

    ########################################
    # Misc
    ########################################
    g_misc = parser.add_argument_group('Misc')
    g_misc.add_argument('--root', default="/ldisk/habara/BayesianSpikeFusion")
    g_misc.add_argument('--print_freq', default=100, type=int)
    g_misc.add_argument('--resume', default="", type=str)  # experiment directory
    g_misc.add_argument('--resume_epoch', default=0, type=int)

    args = parser.parse_args()

    return args

def get_snn_args():
    parser = argparse.ArgumentParser()

    ########################################
    # SNN Setting
    ########################################
    g_snn = parser.add_argument_group('SNN')
    g_snn.add_argument('train_dir')
    g_snn.add_argument('--timestep', default=3000, type=int)
    g_snn.add_argument('--burnin', default=500, type=int)
    g_snn.add_argument('--batch_size', default=500, type=int)

    ########################################
    # ANN2SNN
    ########################################
    g_convert = parser.add_argument_group('convert')
    g_convert.add_argument('--percentile', default=0.999, type=float)
    g_convert.add_argument('--pooling', default="avg", choices=["avg", "rand"])

    ########################################
    # BaysianSpikeFusion
    ########################################
    g_bsf = parser.add_argument_group('BSF')
    g_bsf.add_argument('--init_mem', default=0.0, type=float)
    g_bsf.add_argument('--hps', default="grid", choices=["grid", "emp"], help="hyperparameter search")

    args = parser.parse_args()

    return args

def get_save_dir(args):
    conditions = "{}_{}".format(args.dataset, args.model)
    session_dir = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    save_dir = os.path.join(args.root, conditions, str(args.ic_index), session_dir)

    return conditions, save_dir

def get_save_snn_dir(args):
    conditions = "{}_{}_{}".format(args.dataset, args.model, args.ic_index)
    session_dir = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    save_dir = os.path.join(args.root, conditions, str(args.ic_index), session_dir)

    return conditions, save_dir

def save_args(args, savedir):
    with open(os.path.join(savedir, 'command.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

def load_args(path):
    with open(path, "r") as f:
        exp_args = json.load(f)

    return argparse.Namespace(**exp_args)
