import os 
import numpy as np 
import torch 
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from utils.metrics import test_scikit_ap, test_emotic_vad, get_thresholds

import matplotlib.pyplot as plt


def train_disc(epochs,
              model_path,
              opt, scheduler,
              models, 
              disc_loss, 
              cat_loss_param=1.0, 
              cont_loss_param=0.0, 
              train_length = None, 
              val_length = None,
              train_loader = None,
              val_loader = None,
              device = 'cpu'):

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  min_loss = np.inf
  min_mae = np.inf

  train_loss = list()
  val_loss = list()
  train_mae = list()
  val_mae = list()



  model_context, model_body, model_face, fusion_model = models



  for e in range(epochs):
    running_loss = 0.0
    fusion_model.to(device)
    model_context.to(device)
    model_body.to(device)
    model_face.to(device)
    fusion_model.train()
    model_context.train()
    model_body.train()
    model_face.train()

    train_cat_preds = np.zeros((train_length, 26))
    train_cat_labels = np.zeros((train_length, 26))
    indx = 0



    for images_context, images_body, images_face, labels_cat, labels_cont in tqdm(train_loader, desc="Training Progress", leave=True):

      images_context = images_context.to(device)
      images_body = images_body.to(device)
      images_face = images_face.to(device)
      images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

      labels_cat = labels_cat.to(device)

      opt.zero_grad()

      pred_context = model_context(images_context)
      pred_body = model_body(images_body)
      pred_face = model_face(images_face)



      pred_cat = fusion_model(pred_context, pred_body, pred_face)
      cat_loss_batch = disc_loss(pred_cat, labels_cat)
      loss = (cat_loss_param * cat_loss_batch)
      running_loss += loss.item()

      loss.backward()
      opt.step()

      train_cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
      train_cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
      indx = indx + pred_cat.shape[0]



    if e % 1 == 0:

      print ('epoch = %d training loss = %.4f' %(e, running_loss))

      train_loss.append(running_loss)
      train_cat_preds = train_cat_preds.transpose()
      train_cat_labels = train_cat_labels.transpose()
      train_mae.append(test_scikit_ap(train_cat_preds, train_cat_labels))
      print ('epoch = %d training AP = %.4f' %(e, train_mae[-1]))





    running_loss = 0.0
    fusion_model.eval()
    model_context.eval()
    model_body.eval()
    model_face.eval()

    val_cat_preds = np.zeros((val_length, 26))
    val_cat_labels = np.zeros((val_length, 26))
    indx = 0

    with torch.no_grad():
      for images_context, images_body, images_face, labels_cat, labels_cont in iter(val_loader):

        images_context = images_context.to(device)
        images_body = images_body.to(device)
        images_face = images_face.to(device)
        images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
        labels_cat = labels_cat.to(device)

        pred_context = model_context(images_context)
        pred_body = model_body(images_body)
        pred_face = model_face(images_face)



        pred_cat = fusion_model(pred_context, pred_body, pred_face)
        cat_loss_batch = disc_loss(pred_cat, labels_cat)
        loss =  (cat_loss_param * cat_loss_batch)
        running_loss += loss.item()

        val_cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
        val_cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
        indx = indx + pred_cat.shape[0]

      if e % 1 == 0:

        print ('epoch = %d validation loss = %.4f' %(e, running_loss))
        val_loss.append(running_loss)
        val_cat_preds = val_cat_preds.transpose()
        val_cat_labels = val_cat_labels.transpose()
        val_mae.append(test_scikit_ap(val_cat_preds, val_cat_labels))

        print ('epoch = %d validation AP = %.4f' %(e, val_mae[-1]))

    scheduler.step()
    print('')

    if val_loss[-1] < min_loss:

        min_loss = val_loss[-1]

        # saving models for lowest loss

        print ('saving model at epoch e = %d' %(e))

        fusion_model.to("cpu")
        model_context.to("cpu")
        model_body.to("cpu")
        model_face.to("cpu")




        torch.save(fusion_model, os.path.join(model_path, 'model_fusion.pth'))
        torch.save(model_context, os.path.join(model_path, 'model_context.pth'))
        torch.save(model_body, os.path.join(model_path, 'model_body.pth'))
        torch.save(model_face, os.path.join(model_path, 'model_face.pth'))

  print('completed training')
  return train_loss, val_loss, train_mae, val_mae
  



  #statistic graphic



