from abc import update_abstractmethods
from math import gamma
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser, ArgumentTypeError
from torchsummary import summary
from transformers import DistilBertModel, BertModel
from model.resnet import resnet50V2, resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t
from model.fusion import FusionModel, FusionConcatModel, FusionFullCrossAttentionModel
from dataset.data_loader import load_data, set_normalization_and_transforms
from utils.losses import DiscreteLoss, CrossEtropyLoss, BCEWithLogitsLoss
from utils.training import train_disc
from utils.testing import test_disc



def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise ArgumentTypeError('Boolean value expected.')

def get_arg():
  parser = ArgumentParser()
  parser.add_argument('--save_model', default='./save_model', type=str)
  parser.add_argument('--batch_size', default=26, type=int)
  parser.add_argument('--epochs', default=25, type=int)
  parser.add_argument('--loss', default='L2', type=str, choices=['L2', 'BCE', 'CrossEntropy', 'Huber'])
  parser.add_argument('--swin_model', default=False, type=str2bool)
  parser.add_argument('--path_dataset', default='/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full', type=str)
  parser.add_argument('--learning_rate', default=0.001, type=float)
  parser.add_argument('--weight_decay', default=5e-4, type=float)
  parser.add_argument('--step_size', default=7, type=int)
  parser.add_argument('--gamma', default=0.1, type=float)
  parser.add_argument('--conbine', default='concat', type=str)
  parser.add_argument('--model_text', default='distilbert', choices = ['distilbert', 'bert'], type=str)
  pars = parser.parse_args()
  return pars

def train(pars):

  batch_size = args.batch_size
  data_src = args.path_dataset
  gamma = args.gamma
  conbine = args.conbine
  epochs = args.epochs
  model_path = args.save_model
  isSwinT = args.swin_model
  loss_function = args.loss
  model_text = args.model_text

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



  model_context = resnet50_place365(pretrained = True)
  print(summary(model_context, (3,224,224), device="cpu"))
  model_face = cnn_face(pretrained = True)
  model_body = swin_v2_t(pretrained = True)
  if model_text == "distilbert":
    model_text = DistilBertModel.from_pretrained('distilbert-base-uncased')
  elif model_text == "bert":
    model_text = BertModel.from_pretrained('bert-base-uncased')
  print(model_text)

  num_context_features = list(model_context.children())[-1].in_features
  num_body_features = list(model_body.children())[-1].in_features
  num_face_features = list(model_face.children())[-3].in_features
  num_text_features = model_text.config.dim
  
  print(num_text_features)
  print(num_context_features)
  print(num_body_features)
  print(num_face_features)

  #fusion_model = FusionModel(num_context_features, num_body_features, num_face_features, conbine, isSwinT)
  #fusion_model = FusionConcatModel(num_context_features, num_body_features, num_face_features, num_text_features, isSwinT)
  fusion_model = FusionFullCrossAttentionModel(num_context_features, num_body_features, num_face_features, num_text_features)

  for param in fusion_model.parameters():
    param.requires_grad = True
  for param in model_context.parameters():
    param.requires_grad = False
  for param in model_body.parameters():
    param.requires_grad = False
  for param in model_face.parameters():
    param.requires_grad = False
  for param in model_text.parameters():
    param.requires_grad = False





  opt = optim.AdamW((list(fusion_model.parameters()) + list(model_context.parameters()) + \
                  list(model_body.parameters()) + list(model_face.parameters()) + list(model_text.parameters())), 
                  lr=pars.learning_rate, weight_decay=pars.weight_decay)

  scheduler = StepLR(opt, step_size=7, gamma=gamma)
  if loss_function == "L2":
    disc_loss = DiscreteLoss('dynamic', device)
  elif loss_function == "BCE":
    disc_loss = BCEWithLogitsLoss('dynamic', device)
  elif loss_function == "CrossEntropy":
    disc_loss = CrossEtropyLoss('dynamic', device)


  train_loss, val_loss, train_mae, val_mae = train_disc(epochs, 
            model_path, 
            opt, scheduler, 
            [model_context, model_body, model_face, model_text, fusion_model], 
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



  test_map = test_disc([model_context, model_body, model_face, model_text, fusion_model], device, test_loader, test_length)
  print ('testing mAP=%.4f' %(test_map))

if __name__=='__main__':
  args = get_arg()
  train(args)


  

