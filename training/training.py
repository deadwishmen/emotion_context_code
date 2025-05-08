import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import test_scikit_ap

import matplotlib.pyplot as plt




def train_disc(
    epochs,
    model_path,
    opt,
    scheduler,
    models,
    disc_loss,
    cat_loss_param=1.0,
    cont_loss_param=0.0,
    train_length=None,
    val_length=None,
    train_loader=None,
    val_loader=None,
    device='cpu',
    combine=False,
    choices_model_context=None
):
    """
    Train a multi-modal model with context, body, face, and text inputs.

    Args:
        epochs (int): Number of training epochs.
        model_path (str): Directory to save model checkpoints.
        opt: Optimizer for training.
        scheduler: Learning rate scheduler.
        models (tuple): Tuple of (model_context, model_body, model_face, model_text, fusion_model).
        disc_loss: Loss function for categorical predictions.
        cat_loss_param (float): Weight for categorical loss.
        cont_loss_param (float): Weight for continuous loss.
        train_length (int): Number of training samples.
        val_length (int): Number of validation samples.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device (str): Device to run the models on ('cpu' or 'cuda').
        combine (str or bool): Specifies combination method ('q_former' or False).
        choices_model_context (str): Context model type ('vit' or other).

    Returns:
        tuple: Lists of training loss, validation loss, training AP, and validation AP.
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Initialize tracking variables
    min_loss = np.inf
    min_mae = np.inf
    max_val = 0
    train_loss, val_loss = [], []
    train_mae, val_mae = [], []

    # Unpack models
    model_context, model_body, model_face, model_text, fusion_model = models

    for e in range(epochs):
        # Move models to device
        fusion_model.to(device)
        model_context.to(device)
        model_body.to(device)
        model_face.to(device)
        model_text.to(device)

        # Set models to training mode
        fusion_model.train()
        model_context.train()
        model_body.train()
        model_face.train()
        model_text.train()

        running_loss = 0.0
        train_cat_preds = np.zeros((train_length, 26))
        train_cat_labels = np.zeros((train_length, 26))
        indx = 0

        # Training loop
        for images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont in tqdm(
            train_loader, desc="Training Progress", leave=True
        ):
            # Move data to device
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            tokenizer_text = {key: val.to(device) for key, val in tokenizer_text.items()}
            labels_cat = labels_cat.to(device)

            opt.zero_grad()

            # Forward pass
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)

            if combine == "q_former":
                pred_text = model_text(**tokenizer_text).last_hidden_state
                pred_context = model_context(images_context)[:, 1:]
            elif choices_model_context == "vit":
                pred_context = model_context.get_image_features(pixel_values=images_context)
                pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)
            else:
                pred_context = model_context(images_context)
                pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)

            # Fusion model prediction
            pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
            cat_loss_batch = disc_loss(pred_cat, labels_cat)

            # Total loss
            loss = cat_loss_batch
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            opt.step()

            # Store predictions and labels
            batch_size = pred_cat.shape[0]
            train_cat_preds[indx : indx + batch_size, :] = pred_cat.cpu().numpy()
            train_cat_labels[indx : indx + batch_size, :] = labels_cat.cpu().numpy()
            indx += batch_size

        # Log training metrics
        print(f"Epoch {e}, Training Loss: {running_loss:.4f}")
        train_loss.append(running_loss)
        train_cat_preds = train_cat_preds.transpose()
        train_cat_labels = train_cat_labels.transpose()
        train_ap = test_scikit_ap(train_cat_preds, train_cat_labels)
        train_mae.append(train_ap)
        print(f"Epoch {e}, Training AP: {train_ap:.4f}")

        # Validation loop
        running_loss = 0.0
        fusion_model.eval()
        model_context.eval()
        model_body.eval()
        model_face.eval()
        model_text.eval()

        val_cat_preds = np.zeros((val_length, 26))
        val_cat_labels = np.zeros((val_length, 26))
        indx = 0

        with torch.no_grad():
            for images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont in val_loader:
                # Move data to device
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
                tokenizer_text = {key: val.to(device) for key, val in tokenizer_text.items()}
                labels_cat = labels_cat.to(device)

                # Forward pass
                pred_body = model_body(images_body)
                pred_face = model_face(images_face)

                if combine == "q_former":
                    pred_text = model_text(**tokenizer_text).last_hidden_state
                    pred_context = model_context(images_context)[:, 1:]
                else:
                    pred_text = model_text(**tokenizer_text).last_hidden_state.mean(dim=1)
                    pred_context = model_context(images_context)

                # Fusion model prediction
                pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
                cat_loss_batch = disc_loss(pred_cat, labels_cat)

                # Total loss
                loss = cat_loss_batch
                running_loss += loss.item()

                # Store predictions and labels
                batch_size = pred_cat.shape[0]
                val_cat_preds[indx : indx + batch_size, :] = pred_cat.cpu().numpy()
                val_cat_labels[indx : indx + batch_size, :] = labels_cat.cpu().numpy()
                indx += batch_size

        # Log validation metrics
        print(f"Epoch {e}, Validation Loss: {running_loss:.4f}")
        val_loss.append(running_loss)
        val_cat_preds = val_cat_preds.transpose()
        val_cat_labels = val_cat_labels.transpose()
        val_ap = test_scikit_ap(val_cat_preds, val_cat_labels)
        val_mae.append(val_ap)
        print(f"Epoch {e}, Validation AP: {val_ap:.4f}")

        # Update scheduler
        scheduler.step()

        # Save models if validation AP improves
        if val_mae[-1] > max_val:
            max_val = val_mae[-1]
            print(f"Saving models at epoch {e}")

            # Move models to CPU for saving
            fusion_model.to("cpu")
            model_context.to("cpu")
            model_body.to("cpu")
            model_face.to("cpu")
            model_text.to("cpu")

            # Save model checkpoints
            torch.save(fusion_model, os.path.join(model_path, "model_fusion.pth"))
            torch.save(model_context, os.path.join(model_path, "model_context.pth"))
            torch.save(model_body, os.path.join(model_path, "model_body.pth"))
            torch.save(model_face, os.path.join(model_path, "model_face.pth"))
            torch.save(model_text, os.path.join(model_path, "model_text.pth"))

        print("")

    print("Completed training")

    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot AP
    plt.subplot(1, 2, 2)
    plt.plot(train_mae, label="Training AP")
    plt.plot(val_mae, label="Validation AP")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision")
    plt.title("Training and Validation AP")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_path, "training_metrics.png"))
    plt.show()

    return train_loss, val_loss, train_mae, val_mae