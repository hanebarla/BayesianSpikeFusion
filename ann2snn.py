import os, argparse
import torch

from train_ann import eval
from convert import normalize
from dataset import dataloader_factory
from model_ann import model_factory
from model_snn import SpikingSDN
from argument import get_args, get_snn_args, get_save_dir, load_args
from logger import create_logger
from utils import save_checkpoint

def main():
    snn_args = get_snn_args()

    save_dir = snn_args.train_dir
    args = load_args(os.path.join(save_dir, "command.json"))
    args.batch_size = snn_args.batch_size
    # print(args)
    conditions, _ = get_save_dir(args)

    # create logger
    logger = create_logger(save_dir, conditions, 'ann2snn_log.txt')
    # logger.disabled = True
    logger.info("[Args]: {}".format(str(args.__dict__)))
    logger.info("[SNN Args]: {}".format(str(snn_args.__dict__)))

    # get Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cnt = torch.cuda.device_count()
    logger.info("[Device]: {}".format(device))
    logger.info("[Device Count]: {}".format(device_cnt))

    # get Dataloader
    train_dataloader, val_dataloader, test_dataloader, num_classes, in_shapes, mixup_fn = dataloader_factory(args)
    logger.info("Mixup: {}".format(mixup_fn))

    # get ANN model
    model = model_factory(args, num_classes, in_shapes)
    logger.info("[model]: {}".format(str(model)))

    # load trained ANN
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"))
    logger.info("ANN Accuracy: {}".format(checkpoint["acc"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # ann to snn
    snn = convert(args, snn_args, model, train_dataloader, in_shapes, device, logger)
    logger.info("[SNN]: {}".format(str(snn)))

def convert(args, snn_args, model, dataloader, in_shapes, device, logger):  
    logger.info(" --- start ann2snn conversion --- ")
    snn_path = os.path.join(snn_args.train_dir, "snn_{}.pth".format(snn_args.percentile))
    logger.info("SNN Path: {}".format(snn_path))
    
    if os.path.exists(snn_path):
        if has_bias_in_model(model):
            add_bias_in_conv(model)
        logger.info("SNN model already exists")
        snn_state_dict = torch.load(snn_path)
        snn = SpikingSDN(model, snn_args.batch_size, in_shapes)
        snn.load_state_dict(snn_state_dict["state_dict"])
        return snn

    loss_weight = {
        -1: 1.0,
        args.ic_index: 1.0
    }
    _, before_acc = eval(args, dataloader, model, loss_weight, device)
    logger.info("before conversion, test acc.: {}".format(before_acc))

    result = normalize(model.feature, next(iter(dataloader))[0].to(device))
    for index in model.classifiers.keys():
        normalize(model.classifiers[index], result[1][int(index)], initial_scale_factor=result[0][int(index)])

    _, after_acc = eval(args, dataloader, model, loss_weight, device)
    logger.info("after conversion, test acc.: {}".format(after_acc))
  
    snn = SpikingSDN(model, snn_args.batch_size, in_shapes)
    snn_state_dict = snn.state_dict()
    save_state_dict = {
        "state_dict": snn_state_dict
    }
    save_checkpoint(save_state_dict, False, snn_path, snn_path)

    return snn

def has_bias_in_model(model):
    for layer in model.modules():
        if has_bias(layer):
            return True
    return False

def has_bias(layer):
    return hasattr(layer, "bias") and layer.bias is not None

def add_bias_in_conv(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) and not has_bias(layer):
            layer.bias = torch.nn.Parameter(torch.zeros(layer.out_channels, dtype=layer.weight.dtype, device=layer.weight.device))

if __name__ == "__main__":
    main()
