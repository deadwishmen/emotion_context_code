import os 
import numpy as np 
import torch 
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from utils.metrics import test_scikit_ap, test_emotic_vad, get_thresholds
from captum.attr import FeatureAblation

import matplotlib.pyplot as plt




def test_disc(models, device, data_loader, num_images, conbine = False):

    model_context, model_body, model_face, model_text, fusion_model = models
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))



    with torch.no_grad():

        model_context.to(device)
        model_body.to(device)
        model_face.to(device)
        model_text.to(device)
        fusion_model.to(device)



        model_context.eval()
        model_body.eval()
        model_face.eval()
        model_text.eval()
        fusion_model.eval()

        ablation = FeatureAblation(fusion_model)  # Khởi tạo Feature Ablation
        all_importance = []
        print("Starting testing with Feature Ablation...")

        indx = 0

        print ('starting testing')

        for images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont in tqdm(iter(data_loader), total=len(data_loader)):

            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            tokenizer_text = {key:val.to(device) for key, val in tokenizer_text.items()}



            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)



            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            if conbine == "q_former":
                pred_text = model_text(**tokenizer_text).last_hidden_state
                pred_context = model_context.encoder(images_context)
            else:
                pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)
                pred_context = model_context(images_context)
            pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
            
            attr = ablation.attribute((pred_context, pred_body, pred_face, pred_text), target=0)  
            attr_numpy = [a.cpu().numpy() for a in attr]  
            all_importance.append([a.mean() for a in attr_numpy]) 
            cat_preds[ indx : (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[ indx : (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx = indx + pred_cat.shape[0]


    all_importance = np.array(all_importance)
    feature_importance = all_importance.mean(axis=0)

    feature_groups = ["Context", "Body", "Face", "Text"]
    plt.figure(figsize=(8, 5))
    plt.bar(feature_groups, feature_importance, color=["blue", "orange", "green", "red"])
    plt.xlabel("Feature Groups")
    plt.ylabel("Importance")
    plt.title("Feature Ablation on Test Set")
    plt.savefig("feature_ablation.png")


    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    print ('completed testing')
    ap_mean = test_scikit_ap(cat_preds, cat_labels)
    return ap_mean
