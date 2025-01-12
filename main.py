from abc import update_abstractmethods
from math import gamma
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser
from torchsummary import summary
from model.resnet import resnet50V2, resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t
from model.fusion import FusionModel
from dataset.data_loader import load_data, set_normalization_and_transforms
from utils.losses import DiscreteLoss, CrossEtropyLoss
from utils.metrics import test_scikit_ap, test_emotic_vad, get_thresholds
from utils.training import train_disc
from utils.testing import test_disc

# Tạo batch dữ liệu ngẫu nhiên với kích thước (batch_size, channels, height, width)
batch_size = 4
channels = 3
height = 224
width = 224


def get_arg():
  parser = ArgumentParser()
  parser.add_argument('--save_model', default='./save_model', type=str)
  parser.add_argument('--batch_size', default=26, type=int)
  parser.add_argument('--epochs', default=25, type=int)
  parser.add_argument('--loss', default='L2', type=str)
  parser.add_argument('--model', default='swin', type=str)
  parser.add_argument('--path_dataset', default='/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full', type=str)
  parser.add_argument('--learning_rate', default=0.001, type=float)
  parser.add_argument('--weight_decay', default=5e-4, type=float)
  parser.add_argument('--step_size', default=7, type=int)
  parser.add_argument('--gamma', default=0.1, type=float)
  parser.add_argument('--conbine', default='concat', type=str)
  pars = parser.parse_args()
  return pars

def train(pars):

  batch_size = args.batch_size
  data_src = args.path_dataset
  gamma = args.gamma
  
  conbine = args.conbine
  epochs = args.epochs
  model_path = args.save_model
  isSwinT = False

  if args.model.lower() == 'swin':
    isSwinT = True

  context_norm, body_norm, face_norm, train_transform, test_transform, face_train_transform, face_test_transform = set_normalization_and_transforms(isSwinT)

  train_loader, val_loader, test_loader, cat2ind, ind2cat, train_length, val_length, test_length = load_data(
      data_src, 
      batch_size,
      train_transform,
      test_transform,
      face_train_transform,
      face_test_transform,
      context_norm,
      body_norm,
      face_norm
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  inputs = torch.randn(batch_size, channels, height, width)

  model_context = resnet50_place365(pretrained = True)
  print(summary(model_context, (3,224,224), device="cpu"))
  model_face = cnn_face(pretrained = True)
  # print(summary(model_face, (1, 48,48), device="cpu"))
  model_body = swin_v2_t(pretrained = True)
  # print (summary(model_body, (3,128,128), device="cpu"))


  num_context_features = list(model_context.children())[-1].in_features
  num_body_features = list(model_body.children())[-1].in_features
  num_face_features = list(model_face.children())[-3].in_features
  
  print(num_context_features)
  print(num_body_features)
  print(num_face_features)

  fusion_model = FusionModel(num_context_features, num_body_features, num_face_features, conbine, isSwinT)

  for param in fusion_model.parameters():
    param.requires_grad = True
  for param in model_context.parameters():
    param.requires_grad = False
  for param in model_body.parameters():
    param.requires_grad = False
  for param in model_face.parameters():
    param.requires_grad = False





  opt = optim.AdamW((list(fusion_model.parameters()) + list(model_context.parameters()) + \
                  list(model_body.parameters()) + list(model_face.parameters())), lr=pars.learning_rate, weight_decay=pars.weight_decay)

  scheduler = StepLR(opt, step_size=7, gamma=gamma)
  disc_loss = DiscreteLoss('dynamic', device)


  train_loss, val_loss, train_mae, val_mae = train_disc(epochs, 
            model_path, 
            opt, scheduler, 
            [model_context, model_body, model_face, fusion_model], 
            disc_loss, 
            cat_loss_param=1.0, 
            cont_loss_param=0.0, 
            train_length = train_length, 
            val_length = val_length,
            train_loader = train_loader,
            val_loader = val_loader,
            device = device
            )

  f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize = (15, 10))
  f.suptitle('Multi-Branch Network for Imagery Emotion Prediction')
  ax1.plot(range(0,len(train_loss)),train_loss, color='Blue')
  ax2.plot(range(0,len(val_loss)),val_loss, color='Red')
  ax1.legend(['train loss'])
  ax2.legend(['val loss'])



  ax3.plot(range(0,len(train_mae)),train_mae, color='Blue')
  ax4.plot(range(0,len(val_mae)),val_mae, color='Red')
  ax3.legend(['train mAP'])
  ax4.legend(['val mAP'])



  test_map = test_disc([model_context, model_body, model_face, fusion_model], device, test_loader, test_length)
  print ('testing mAP=%.4f' %(test_map))

if __name__=='__main__':
  args = get_arg()
  train(args)


  

