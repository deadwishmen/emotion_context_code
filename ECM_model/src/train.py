import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import test_scikit_ap, test_emotic_vad

def train_disc(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, device, train_loader, val_loader, train_dataset, val_dataset, cat_loss_param=1.0, cont_loss_param=0.0):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    min_loss = np.inf
    train_loss, val_loss, train_mae, val_mae = [], [], [], []
    model_context, model_body, model_face, fusion_model = models
    isBFER = False  # Adjust based on model configuration

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

        train_cat_preds = np.zeros((len(train_dataset), 26))
        train_cat_labels = np.zeros((len(train_dataset), 26))
        indx = 0

        for images_context, images_body, images_face, labels_cat, labels_cont in train_loader:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not isBFER:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            labels_cat = labels_cat.to(device)

            opt.zero_grad()
            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cat = fusion_model(pred_context, pred_body, pred_face)
            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            loss = cat_loss_param * cat_loss_batch
            running_loss += loss.item()

            loss.backward()
            opt.step()

            train_cat_preds[indx: (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
            train_cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
            indx += pred_cat.shape[0]

        print(f'epoch = {e} training loss = {running_loss:.4f}')
        train_loss.append(running_loss)
        train_cat_preds = train_cat_preds.transpose()
        train_cat_labels = train_cat_labels.transpose()
        train_mae.append(test_scikit_ap(train_cat_preds, train_cat_labels))
        print(f'epoch = {e} training AP = {train_mae[-1]:.4f}')

        running_loss = 0.0
        fusion_model.eval()
        model_context.eval()
        model_body.eval()
        model_face.eval()

        val_cat_preds = np.zeros((len(val_dataset), 26))
        val_cat_labels = np.zeros((len(val_dataset), 26))
        indx = 0
        with torch.no_grad():
            for images_context, images_body, images_face, labels_cat, labels_cont in val_loader:
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
                cat_loss_batch = disc_loss(pred_cat, labels_cat)
                loss = cat_loss_param * cat_loss_batch
                running_loss += loss.item()

                val_cat_preds[indx: (indx + pred_cat.shape[0]), :] = pred_cat.to("cpu").data.numpy()
                val_cat_labels[indx: (indx + labels_cat.shape[0]), :] = labels_cat.to("cpu").data.numpy()
                indx += pred_cat.shape[0]

        print(f'epoch = {e} validation loss = {running_loss:.4f}')
        val_loss.append(running_loss)
        val_cat_preds = val_cat_preds.transpose()
        val_cat_labels = val_cat_labels.transpose()
        val_mae.append(test_scikit_ap(val_cat_preds, val_cat_labels))
        print(f'epoch = {e} validation AP = {val_mae[-1]:.4f}')

        scheduler.step()
        if val_loss[-1] < min_loss:
            min_loss = val_loss[-1]
            print(f'saving model at epoch e = {e}')
            fusion_model.to("cpu")
            model_context.to("cpu")
            model_body.to("cpu")
            model_face.to("cpu")
            torch.save(fusion_model, os.path.join(model_path, 'model_fusion.pth'))
            torch.save(model_context, os.path.join(model_path, 'model_context.pth'))
            torch.save(model_body, os.path.join(model_path, 'model_body.pth'))
            torch.save(model_face, os.path.join(model_path, 'model_face.pth'))

    print('completed training')
    plot_training_stats(train_loss, val_loss, train_mae, val_mae, 'Discrete')

def train_cont(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, device, train_loader, val_loader, train_dataset, val_dataset, cat_loss_param=0, cont_loss_param=1.0):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    min_loss = np.inf
    train_loss, val_loss, train_mae, val_mae = [], [], [], []
    model_context, model_body, model_face, fusion_model = models
    isBFER = False

    for e in range(epochs):
        running_loss = 0.0
        model_context.to(device)
        model_body.to(device)
        model_face.to(device)
        fusion_model.to(device)
        model_context.train()
        model_body.train()
        model_face.train()
        fusion_model.train()

        train_cont_preds = np.zeros((len(train_dataset), 3))
        train_cont_labels = np.zeros((len(train_dataset), 3))
        indx = 0

        for images_context, images_body, images_face, labels_cat, labels_cont in train_loader:
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device)
            if not isBFER:
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            labels_cont = labels_cont.to(device)

            opt.zero_grad()
            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_cont = fusion_model(pred_context, pred_body, pred_face)
            cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
            loss = cont_loss_param * cont_loss_batch
            running_loss += loss.item()
            loss.backward()
            opt.step()

            train_cont_preds[indx: (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
            train_cont_labels[indx: (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
            indx += pred_cont.shape[0]

        print(f'epoch = {e} training loss = {running_loss:.4f}')
        train_loss.append(running_loss)
        train_cont_preds = train_cont_preds.transpose()
        train_cont_labels = train_cont_labels.transpose()
        train_mae.append(test_emotic_vad(train_cont_preds, train_cont_labels))
        print(f'epoch = {e} training MAE = {train_mae[-1]:.4f}')

        running_loss = 0.0
        model_context.eval()
        model_body.eval()
        model_face.eval()
        fusion_model.eval()

        val_cont_preds = np.zeros((len(val_dataset), 3))
        val_cont_labels = np.zeros((len(val_dataset), 3))
        indx = 0

        with torch.no_grad():
            for images_context, images_body, images_face, labels_cat, labels_cont in val_loader:
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                images_face = images_face.to(device)
                if not isBFER:
                    images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
                labels_cont = labels_cont.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_face = model_face(images_face)
                pred_cont = fusion_model(pred_context, pred_body, pred_face)
                cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                loss = cont_loss_param * cont_loss_batch
                running_loss += loss.item()

                val_cont_preds[indx: (indx + pred_cont.shape[0]), :] = pred_cont.to("cpu").data.numpy() * 10
                val_cont_labels[indx: (indx + labels_cont.shape[0]), :] = labels_cont.to("cpu").data.numpy() * 10
                indx += pred_cont.shape[0]

        print(f'epoch = {e} validation loss = {running_loss:.4f}')
        val_loss.append(running_loss)
        val_cont_preds = val_cont_preds.transpose()
        val_cont_labels = val_cont_labels.transpose()
        val_mae.append(test_emotic_vad(val_cont_preds, val_cont_labels))
        print(f'epoch = {e} val MAE = {val_mae[-1]:.4f}')
        scheduler.step()

        if val_loss[-1] < min_loss:
            min_loss = val_loss[-1]
            print(f'saving model at epoch e = {e}')
            fusion_model.to("cpu")
            model_context.to("cpu")
            model_body.to("cpu")
            model_face.to("cpu")
            torch.save(fusion_model, os.path.join(model_path, 'model_fusion.pth'))
            torch.save(model_context, os.path.join(model_path, 'model_context.pth'))
            torch.save(model_body, os.path.join(model_path, 'model_body.pth'))
            torch.save(model_face, os.path.join(model_path, 'model_face.pth'))

    print('completed training')
    plot_training_stats(train_loss, val_loss, train_mae, val_mae, 'Continuous')

def plot_training_stats(train_loss, val_loss, train_mae, val_mae, prediction_type):
    f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 10))
    f.suptitle(f'Multi-Branch Network for Imagery Emotion Prediction ({prediction_type})')
    ax1.plot(range(len(train_loss)), train_loss, color='Blue')
    ax2.plot(range(len(val_loss)), val_loss, color='Red')
    ax1.legend(['train loss'])
    ax2.legend(['val loss'])
    ax3.plot(range(len(train_mae)), train_mae, color='Blue')
    ax4.plot(range(len(val_mae)), val_mae, color='Red')
    ax3.legend([f'train {"MAE" if prediction_type == "Continuous" else "mAP"}'])
    ax4.legend([f'val {"MAE" if prediction_type == "Continuous" else "mAP"}'])
    plt.show()

def train(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, device, train_loader, val_loader, train_dataset, val_dataset, isVADPrediction):
    if isVADPrediction:
        train_cont(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, device, train_loader, val_loader, train_dataset, val_dataset, cat_loss_param=1.0, cont_loss_param=0.0)
    else:
        train_disc(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, device, train_loader, val_loader, train_dataset, val_dataset, cat_loss_param=1.0, cont_loss_param=0.0)