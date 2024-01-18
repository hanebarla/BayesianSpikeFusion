import os
import time

import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataset import dataloader_factory
from model_ann import model_factory
from argument import get_args, get_save_dir, save_args, load_args
from logger import create_logger
from utils import save_checkpoint, plot_train_curv, get_time


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.opt_moment,
                                    nesterov=True,
                                    weight_decay=args.opt_wd)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      eps=args.opt_eps,
                                      betas=args.opt_betas,
                                      lr=args.lr,
                                      weight_decay=args.opt_wd)
    else:
        raise ValueError("No such optimizer")
    
    scheduler = None
    if args.lr_sch == "cosine":
        scheduler = CosineLRScheduler(optimizer, t_initial=args.lr_t_initial,
                                      lr_min=args.lr_min, warmup_t=args.warmup_t, 
                                      warmup_lr_init=args.warmup_lr_init, warmup_prefix=True)
    elif args.lr_sch == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=args.gamma)
    
    return optimizer, scheduler

def main():
    # get args
    args = get_args()

    if args.resume == "":
        # create directory to save experiments results
        conditions, save_dir = get_save_dir(args)
        os.makedirs(save_dir, exist_ok=False)
        save_args(args, save_dir)
    else:
        old_args = args
        save_dir = args.resume
        args = load_args(os.path.join(save_dir, "command.json"))
        conditions, _ = get_save_dir(args)
        checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"))
        args.resume_epoch = checkpoint['epoch']
        # print(args.resume, ", ", old_args.resume)
        # print(args.resume_epoch)
        # raise ValueError
        args.resume = old_args.resume

    # create logger
    logger = create_logger(save_dir, conditions, 'train_log.txt')
    if args.resume != "":
        logger.info("Train Resumed")
    # logger.disabled = True
    logger.info("[Args]: {}".format(str(args.__dict__)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cnt = torch.cuda.device_count()
    logger.info("[Device]: {}".format(device))
    logger.info("[Device Count]: {}".format(device_cnt))

    # dataloader
    train_dataloader, val_dataloader, test_dataloader, num_classes, in_shapes, mixup_fn = dataloader_factory(args)
    logger.info("Mixup: {}".format(mixup_fn))

    # get model
    model = model_factory(args, num_classes, in_shapes)
    if device_cnt > 1:
        model = nn.DataParallel(model)
    if args.resume != "":
        model.load_state_dict(checkpoint["state_dict"])
    logger.info("[model]: {}".format(str(model)))
    model.to(device)

    # caoculate loss_weight_goal
    shape = in_shapes
    total_flops = []
    sum = 0
    logger.info("flops, shape")
    if isinstance(model, torch.nn.DataParallel):
        for module in model.module.feature:
            flops, shape = module.flops(shape)
            sum += flops
            total_flops.append(sum)
            logger.info("{}, {}".format(flops, shape))
    else:
        for module in model.feature:
            flops, shape = module.flops(shape)
            sum += flops
            total_flops.append(sum)
            logger.info("{}, {}".format(flops, shape))
    if args.ic_index is not None:
        loss_weight_goal = total_flops[args.ic_index] / total_flops[-1]

    # train
    train(args, train_dataloader, val_dataloader, test_dataloader, mixup_fn, loss_weight_goal, model, device, save_dir, logger)

def train(args, train_dataloader, val_dataloader, test_dataloader, mixup_fn, loss_weight_goal, model, device, save_dir, logger):
    # cuda benchmark True
    torch.backends.cudnn.benchmark = True

    train_losses, train_accs = [], {args.ic_index: [], -1: []}
    val_losses, val_accs = [], {args.ic_index: [], -1: []}

    optimizer, scheduler = get_optimizer(args, model)
    logger.info("optimizer: {}, scheduler: {}".format(optimizer, scheduler))

    logger.info(' --- start train ANN --- ')

    if args.mixup > 0:
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    logger.info("criterion: {}".format(criterion))

    loss_weight = {
        args.ic_index: 0.01,
        -1: 1.0
    }

    start_time = time.time()
    best_acc = 0.0
    for e in range(args.epochs):
        if e < args.resume_epoch:
            continue
        is_best = False

        logger.info("=== [{} / {}] epochs ===".format(e+1, args.epochs))
        model.train()
        train_loss, train_acc = train_epoch(args, train_dataloader, mixup_fn, 
                                            model, loss_weight, criterion, 
                                            optimizer, device, start_time, logger)
        train_losses.append(train_loss)
        for index in train_acc.keys():
            train_accs[index].append(train_acc[index])

        model.eval()
        if val_dataloader is not None:
            val_loss, val_acc = eval(args, val_dataloader, model, loss_weight, device)
            val_losses.append(val_loss)
            for index in val_acc.keys():
                val_accs[index].append(val_acc[index])

            now_time = time.time()
            logger.info("Val loss: {}, Acc (M): {}, Acc (F): {}, Time: {}".format(val_loss, 
                                                                                  val_acc[args.ic_index], 
                                                                                  val_acc[-1], 
                                                                                  get_time(start_time, now_time)))

            is_best = val_acc[-1] > best_acc
            if is_best:
                best_acc = val_acc[-1]
        else:
            is_best = train_acc[-1] > best_acc
            if is_best:
                best_acc = train_acc[-1]

        save_state = {
            "state_dict": model.state_dict(),
            'epoch': e,
            'acc': best_acc
        }
        save_checkpoint(save_state, 
                        is_best, 
                        filename=os.path.join(save_dir, "checkpoint.pth"),
                        bestname=os.path.join(save_dir, "model-best.pth"))

        if args.lr_sch == 'cosine':
            scheduler.step(e)
        elif args.lr_sch == 'multistep':
            scheduler.step()
        if args.ic_index is not None and args.ic_index != -1:
            loss_weight[args.ic_index] = 0.01 + e / args.epochs * (loss_weight_goal-0.01)

    model.eval()
    test_loss, test_acc = eval(args, test_dataloader, model, loss_weight, device)
    now_time = time.time()
    logger.info("Test loss: {}, Acc (M): {}, Acc (F): {}, Time: {}".format(test_loss, 
                                                                           test_acc[args.ic_index], 
                                                                           test_acc[-1], 
                                                                           get_time(start_time, now_time)))

    with open(os.path.join(save_dir, "ann_acc.csv"), "w") as f:
        for k, v in test_acc.items():
            f.write(f"{k},{v}\n")
    plot_train_curv([train_losses, val_losses], [train_accs, val_accs], args.ic_index, save_dir)


def train_epoch(args, train_dataloader, mixup_fn, model, loss_weight, criterion, optimizer, device, start_time, logger):
    dlengs = len(train_dataloader.dataset)
    train_loss = 0
    train_acc = {args.ic_index: 0, -1: 0}

    scaler = torch.cuda.amp.GradScaler()

    for i, (images, labels) in enumerate(train_dataloader):
        # print(images.size())
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # Forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = 0.0
            for index, out in outputs.items():
                loss += loss_weight[index] * criterion(out, labels)

        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        train_loss += loss.item()

        acc_tensors = {}
        for index, out in outputs.items():
            output_argmax = torch.argmax(out, dim=1)
            if mixup_fn is not None:
                labels_argsmax = torch.argmax(labels, dim=1)
                acc_tensor = torch.zeros_like(labels_argsmax)
                acc_tensor[output_argmax==labels_argsmax] = 1
            else:
                acc_tensor = torch.zeros_like(labels)
                acc_tensor[output_argmax==labels] = 1
            acc_nums = acc_tensor.sum().item()
            acc_tensors[index] = acc_nums / images.size(0)
            train_acc[index] += acc_nums

        if (i % args.print_freq) == 0:
            now_time = time.time()
            logger.info('[{} / {}] loss: {}, Acc (M) :{}, Acc (F): {}, Time: {}'.format(min(args.batch_size*i, dlengs), 
                                                                                            dlengs,
                                                                                            loss.item(), 
                                                                                            acc_tensors[args.ic_index],
                                                                                            acc_tensors[-1],
                                                                                            get_time(start_time, now_time)))

    train_loss /= dlengs
    for index, out in outputs.items():
        train_acc[index] /= dlengs

    return train_loss, train_acc

@torch.no_grad()
def eval(args, test_dataloader, model, loss_weight, device):
    dlengs = len(test_dataloader.dataset)
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_acc = {args.ic_index: 0, -1: 0}

    for i, data in enumerate(test_dataloader):
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = 0.0
        for index, out in outputs.items():
            output_argmax = torch.argmax(out, dim=1)
            test_acc[index] += torch.sum(output_argmax==labels).item()
            loss += loss_weight[index] * criterion(out, labels)

        test_loss += loss.item()

    test_loss /= dlengs
    for index in test_acc.keys():
        test_acc[index] /= dlengs

    return test_loss, test_acc


if __name__ == "__main__":
    main()
