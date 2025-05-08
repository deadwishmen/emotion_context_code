import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import test_scikit_ap, get_thresholds
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt


def test_disc(models, device, data_loader, num_images, combine="default", xai=False):
    """
    Test the model ensemble on a dataset and optionally perform feature ablation.

    Args:
        models: Tuple of (model_context, model_body, model_face, model_text, fusion_model)
        device: Torch device (e.g., 'cuda' or 'cpu')
        data_loader: DataLoader for the test dataset
        num_images: Total number of images in the dataset
        combine: Combination method ('q_former' or 'default')
        xai: Whether to perform feature ablation (default: False)

    Returns:
        float: Mean average precision score
    """
    model_context, model_body, model_face, model_text, fusion_model = models
    cat_preds = np.zeros((num_images, 26))
    cat_labels = np.zeros((num_images, 26))

    # Move models to device and set to evaluation mode
    for model in models:
        model.to(device)
        model.eval()

    ablation = FeatureAblation(fusion_model) if xai else None
    all_importance = []

    idx = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont = batch

            # Move inputs to device
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = images_face.to(device).mean(dim=1, keepdim=True)
            tokenizer_text = {k: v.to(device) for k, v in tokenizer_text.items()}
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            # Forward pass
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_text = model_text(**tokenizer_text).last_hidden_state
            pred_context = model_context(images_context)

            if combine == "q_former":
                pred_text = pred_text
                pred_context = pred_context[:, 1:]
            else:
                pred_text = pred_text.mean(dim=1)

            pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)

            # Feature ablation if enabled
            if xai:
                attr = ablation.attribute((pred_context, pred_body, pred_face, pred_text), target=0)
                attr_numpy = [a.cpu().numpy().mean() for a in attr]
                all_importance.append(attr_numpy)

            # Store predictions and labels
            batch_size = pred_cat.shape[0]
            cat_preds[idx:idx + batch_size] = pred_cat.cpu().numpy()
            cat_labels[idx:idx + batch_size] = labels_cat.cpu().numpy()
            idx += batch_size

    # Save thresholds
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join('/', 'thresholds.npy'), thresholds)

    # Visualize feature importance if XAI is enabled
    if xai:
        all_importance = np.array(all_importance).mean(axis=0)
        feature_groups = ["Context", "Body", "Face", "Text"]
        plt.figure(figsize=(8, 5))
        plt.bar(feature_groups, all_importance, color=["blue", "orange", "green", "red"])
        plt.xlabel("Feature Groups")
        plt.ylabel("Importance")
        plt.title("Feature Ablation on Test Set")
        plt.savefig("feature_ablation.png")
        plt.close()

    # Compute and return average precision
    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    return test_scikit_ap(cat_preds, cat_labels)


def visualize_inputs_and_predictions(
    images_context, images_body, images_face, tokenizer_text, 
    cat_preds, cat_labels, batch_idx, tokenizer=None
):
    """
    Visualizes input modalities and model predictions vs. ground truth labels.

    Args:
        images_context: Tensor of context images (batch_size, C, H, W)
        images_body: Tensor of body images (batch_size, C, H, W)
        images_face: Tensor of face images (batch_size, C, H, W)
        tokenizer_text: Dict of tokenized text inputs
        cat_preds: Numpy array of predictions (batch_size, 26)
        cat_labels: Numpy array of ground truth labels (batch_size, 26)
        batch_idx: Current batch index for naming output files
        tokenizer: Optional tokenizer to decode text (e.g., BERT tokenizer)
    """
    sample_idx = 0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Helper function to plot image
    def plot_image(ax, img, title):
        img = img[sample_idx].cpu().numpy().transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = img.squeeze()
            ax.imshow(img, cmap='gray')
        else:
            img = (img - img.min()) / (img.max() - img.min())
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    # Plot images
    plot_image(axes[0, 0], images_context, "Context Image")
    plot_image(axes[0, 1], images_body, "Body Image")
    plot_image(axes[1, 0], images_face, "Face Image")

    # Display text input
    axes[1, 1].axis('off')
    if tokenizer is not None and 'input_ids' in tokenizer_text:
        text_ids = tokenizer_text['input_ids'][sample_idx].cpu().numpy()
        decoded_text = tokenizer.decode(text_ids, skip_special_tokens=True)
    else:
        decoded_text = "Text input not decodable (tokenizer unavailable)"
    axes[1, 1].text(0.1, 0.5, decoded_text, wrap=True, fontsize=10, verticalalignment='center')
    axes[1, 1].set_title("Text Input")

    plt.tight_layout()
    plt.savefig(f"input_visualization_batch_{batch_idx}.png")
    plt.close()

    # Visualize predictions vs. labels
    plt.figure(figsize=(10, 6))
    emotion_categories = [f"Cat_{i}" for i in range(26)]
    pred = cat_preds[sample_idx]
    label = cat_labels[sample_idx]

    x = np.arange(len(emotion_categories))
    width = 0.35

    plt.bar(x - width/2, pred, width, label='Predictions', color='blue', alpha=0.6)
    plt.bar(x + width/2, label, width, label='Ground Truth', color='orange', alpha=0.6)

    plt.xlabel("Emotion Categories")
    plt.ylabel("Probability / Label")
    plt.title("Predictions vs. Ground Truth")
    plt.xticks(x, emotion_categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"prediction_visualization_batch_{batch_idx}.png")
    plt.close()