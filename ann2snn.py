import os, argparse
import torch

from train_ann import eval_ann
from convert import normalize
from dataset import dataset_factory
from model_ann import model_factory
from model_snn import SpikingSDN

parser = argparse.ArgumentParser()
parser.add_argument('model', default='vgg11', type=str)
parser.add_argument('dataset', default='cifar10', type=str)
parser.add_argument('--gpu', type=str)
parser.add_argument('--ic_index', type=int, default=-1)
args = parser.parse_args()

def ann2snn(ic_index, work_dir, dataset_dir):
  device = torch.device('cuda' if torch.cuda.is_available() and "resnet" not in args.model else 'cpu')
  print(device)

  print()
  print(" --- start ann2snn conversion --- ")

  batch_size = 500
  train_dataloader, test_dataloader, num_classes, input_shape = dataset_factory(args.dataset, dataset_dir, batch_size, normalize=False, augument=True)
  
  model = model_factory(args.model, ic_index, activation="relu", num_classes=num_classes, input_channel=input_shape[0])
  model.load_state_dict(torch.load(os.path.join(work_dir, "ann.pth")))

  print("before conversion, test acc.:", eval_ann(model, test_dataloader, device=device))

  result = normalize(model.feature, next(iter(train_dataloader))[0].to(device))
  # print(result[0])

  for index in model.classifiers.keys():
      normalize(model.classifiers[index], result[1][int(index)], initial_scale_factor=result[0][int(index)])

  print("after conversion, test acc.:", eval_ann(model, test_dataloader, device=device))

  spiking_model = SpikingSDN(model, batch_size, input_shape)
  torch.save(spiking_model.cpu().state_dict(), os.path.join(work_dir, "snn.pth"))

  print("spiiking model")
  print(spiking_model)

if __name__ == "__main__":
  if args.gpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

  work_dir = os.path.join(os.environ["DATADIR"], "models", "sdn-{}_{}".format(args.model, args.dataset), "{}_integrated".format(args.ic_index))
  # os.makedirs(work_dir, exist_ok=True)

  dataset_dir = os.path.join(os.environ["DATADIR"], "datasets")

  ann2snn(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir)

