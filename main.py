from abc import update_abstractmethods
from math import gamma
import torch.nn as nn
import torch
import timm 
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser, ArgumentTypeError
from torchsummary import summary
from transformers import AutoModel
from model.resnet import resnet50V2, resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b
from model.vit import vit_b_16
from model.fusion import FusionModel, FusionConcatModel, TransformerFusionModel, DualPathAttentionFusion, QFormer
from dataset.data_loader import load_data, set_normalization_and_transforms
from utils.losses import DiscreteLoss, BCEWithLogitsLoss, FocalLoss
from training.training import train_disc
from training.testing import test_disc
from utils.predict import predict_and_show
from transformers import CLIPModel
import os
import pandas as pd


class_names = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion", "Confidence",
    "Disapproval", "Disconnection", "Disquietment", "Doubt/Confusion", "Embarrassment",
    "Engagement", "Esteem", "Excitement", "Fatigue", "Fear", "Happiness", "Pain",
    "Peace", "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy",
    "Yearning"
]

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
  parser.add_argument('--loss', default='L2', type=str, choices=['L2', 'BCE', 'CrossEntropy', 'Huber', 'FocalLoss'])
  parser.add_argument('--model_body', default='swin-t', type=str, choices=['swin-t', 'swin-s', 'swin-b', 'swin-l', 'resnet', 'vit'])
  parser.add_argument('--model_context', default='resnet', type=str, choices=['resnet', 'vit'])
  parser.add_argument('--path_dataset', default='/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full', type=str)
  parser.add_argument('--learning_rate', default=0.001, type=float)
  parser.add_argument('--weight_decay', default=5e-4, type=float)
  parser.add_argument('--step_size', default=7, type=int)
  parser.add_argument('--gamma', default=0.1, type=float)
  parser.add_argument('--predict', default=False,type=str2bool)
  parser.add_argument('--xai', type=str2bool)
  parser.add_argument('--conbine', default='concat',choices=['concat', 'sum', 'avg', 'q_former' ,'transformer', 'adaptive', 'attention'], type=str)
  parser.add_argument('--model_text', default='distilbert', choices = ['distilbert', 'bert', 'roberta', 'deberta'], type=str)
  pars = parser.parse_args()
  return pars

def train(pars):

  batch_size = args.batch_size
  data_src = args.path_dataset
  gamma = args.gamma
  conbine = args.conbine
  epochs = args.epochs
  choices_model_body = args.model_body
  choices_model_context = args.model_context
  model_path = args.save_model
  loss_function = args.loss
  model_text = args.model_text
  num_context_features = 768
  xai = args.xai
  context_norm, body_norm, face_norm, train_transform, test_transform, face_train_transform, face_test_transform = set_normalization_and_transforms(choices_model_body)

  train_loader, val_loader, test_loader, cat2ind, ind2cat, train_length, val_length, test_length = load_data(
      data_src,
      batch_size,
      train_transform,
      test_transform,
      face_train_transform,
      face_test_transform,
      context_norm,
      body_norm,
      face_norm,
      model_text
  )

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model_dict = {
    "swin-t": swin_v2_t,
    "swin-s": swin_v2_s,
    "swin-b": swin_v2_b,
    "vit": vit_b_16
  }
  if conbine == "q_former":
    model_context = vit_b_16(pretrained = True)
  elif choices_model_context == "resnet": 
    model_context = resnet50_place365(pretrained = True)
    print(summary(model_context, (3,224,224), device="cpu"))
    num_context_features = list(model_context.children())[-1].in_features
  model_face = cnn_face(pretrained = True)
  
  model_body = model_dict.get(choices_model_body, None)
  if model_body:
      model_body = model_body(pretrained=True)
  
  if model_text == "distilbert":
    model_text = AutoModel.from_pretrained('distilbert-base-uncased')
  elif model_text == "bert":
    model_text = AutoModel.from_pretrained('bert-base-uncased')
  elif model_text == "roberta":
    model_text = AutoModel.from_pretrained('roberta-base')
  elif model_text == "deberta":
    model_text = AutoModel.from_pretrained('microsoft/deberta-v2-xlarge')
  print(model_text)

  print(model_body)
  
  
  last_layer = list(model_body.children())[-1]  # Lấy lớp cuối cùng

  # Nếu lớp cuối cùng là Sequential, lấy lớp con cuối cùng trong nó
  if isinstance(last_layer, torch.nn.Sequential):
      last_layer = list(last_layer.children())[-1]

  # Kiểm tra nếu nó là Linear thì mới lấy in_features
  if hasattr(last_layer, 'in_features'):
      num_body_features = last_layer.in_features
  else:
      raise ValueError("The last layer has no in_features. Need to recheck the model.")

  # num_body_features = list(model_body.children())[-1].in_features
  num_face_features = list(model_face.children())[-3].in_features
  num_text_features = model_text.config.hidden_size
  
  print(num_text_features)
  print(num_context_features)
  print(num_body_features)
  print(num_face_features)

  if conbine == "concat":
    fusion_model = FusionConcatModel(num_context_features, num_body_features, num_face_features, num_text_features)
  elif conbine == "sum":
    fusion_model = FusionModel(num_context_features, num_body_features, num_face_features, num_text_features, conbine)
  elif conbine == "avg":
    fusion_model = FusionModel(num_context_features, num_body_features, num_face_features, num_text_features, conbine)
  elif conbine == "transformer":
    fusion_model = TransformerFusionModel(num_context_features, num_body_features, num_face_features, num_text_features) 
  elif conbine == "attention":
    fusion_model = DualPathAttentionFusion(num_context_features, num_body_features, num_face_features, num_text_features)
  elif conbine == "q_former":
    fusion_model = QFormer(num_context_features, num_body_features, num_face_features, num_text_features) 

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


  # parm = (list(fusion_model.parameters()) + list(model_context.parameters()) + \
  #                 list(model_body.parameters()) + list(model_face.parameters()) + list(model_text.parameters()))
  parm = (list(fusion_model.parameters()) + list(model_text.parameters()))

  opt = optim.AdamW(parm, 
                  lr=pars.learning_rate, weight_decay=pars.weight_decay)

  scheduler = StepLR(opt, step_size=7, gamma=gamma)
  if loss_function == "L2":
    disc_loss = DiscreteLoss('dynamic', device)
  elif loss_function == "BCE":
    disc_loss = BCEWithLogitsLoss('dynamic', device)
  elif loss_function == "FocalLoss":
    disc_loss = FocalLoss(gamma=2.0, alpha=None, weight_type='mean', device=device)



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
                device = device,
                conbine = conbine,
                choices_model_context = choices_model_context
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


    model_context = torch.load(os.path.join(model_path, 'model_context.pth'), weights_only=False)
    model_body = torch.load(os.path.join(model_path, 'model_body.pth'), weights_only=False)
    model_face = torch.load(os.path.join(model_path, 'model_face.pth'), weights_only=False)
    fusion_model = torch.load(os.path.join(model_path, 'model_fusion.pth'), weights_only=False)

    model_context.eval()
    model_body.eval()
    model_face.eval()
    fusion_model.eval()

    print ('completed cell')

    test_map = test_disc([model_context, model_body, model_face, model_text, fusion_model], device, val_loader, val_length, conbine = conbine, xai = xai)
    print ('testing mAP=%.4f' %(test_map))
    if pars.predict:

        # Đọc file CSV
        path_dataset = "/kaggle/input/emotion-torch/context_dataset/context_dataset/test.csv"
        df = pd.read_csv(path_dataset)

        # Lấy danh sách các đoạn văn bản
        sentences = df['Output'].tolist()
        predict_and_show(
            [model_context, model_body, model_face, model_text, fusion_model],
            device,
            test_loader,
            sentences=sentences,
            num_samples=test_length,  # hoặc số mẫu bạn muốn hiển thị
            class_names=class_names,  # cung cấp danh sách tên lớp nếu có
            conbine=conbine,  # giá trị của conbine, ví dụ: "q_former" hoặc False
            thresholds_path='/kaggle/working/model_text/thresholds.npy'
        )
        plt.show()
if __name__=='__main__':
  args = get_arg()
  train(args)

 
