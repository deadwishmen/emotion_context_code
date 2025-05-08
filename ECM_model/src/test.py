import torch
import numpy as np
from utils import test_scikit_ap, test_emotic_vad

def test_disc(models, device, data_loader, num_images):
    model_context, model_body, model_face, fusion_model = models
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))
    isBFER = False

    with torch.no_grad():
        model_context.to(device)
        model_body.to(device)
        model_face.to(device)
        fusion_model.to(device)
        model_context.eval()
        model_body.eval()
        model_face.eval()
        fusion_model.eval()

        indx = 0
        print('starting testing')
        for images_context, images_body, images_face, labels_cat, labels_cont in data_loader:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not isBFER:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            labels_cat = labels_cat.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cat = fusion_model(pred_context, pred_body, pred_face)

            cat_preds[indx: (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx += pred_cat.shape[0]

    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    print('completed testing')
    ap_mean = test_scikit_ap(cat_preds, cat_labels)
    return ap_mean

def test_cont(models, device, data_loader, num_images):
    model_context, model_body, model_face, fusion_model = models
    cont_preds = np.zeros((num_images, 3))
    cont_labels = np.zeros((num_images, 3))
    isBFER = False

    with torch.no_grad():
        model_context.to(device)
        model_body.to(device)
        model_face.to(device)
        fusion_model.to(device)
        model_context.eval()
        model_body.eval()
        model_face.eval()
        fusion_model.eval()

        indx = 0
        print('starting testing')
        for images_context, images_body, images_face, labels_cat, labels_cont in data_loader:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not isBFER:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cont = fusion_model(pred_context, pred_body, pred_face)

            cont_preds[indx: (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
            cont_labels[indx: (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
            indx += pred_cont.shape[0]

    cont_preds = cont_preds.transpose()
    cont_labels = cont_labels.transpose()
    print('completed testing')
    vad_mean = test_emotic_vad(cont_preds, cont_labels)
    return vad_mean