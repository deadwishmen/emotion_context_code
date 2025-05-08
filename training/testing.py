import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import test_scikit_ap, get_thresholds
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt
import os


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
    np.save(os.path.join(os.getcwd(), 'thresholds.npy'), thresholds)

    # Visualize feature importance if XAI is enabled
    if xai:
        all_importance = np.array(all_importance).mean(axis=0)
        feature_groups = ["Context", "Body", "Face", "Text"]
        plt.figure(figsize=(8, 5))
        plt.bar(feature_groups, all_importance, color=["blue", "orange", "green", "red"])
        plt.xlabel("Feature Groups")
        plt.ylabel("Importance")
        plt.title("Feature Ablation on Test Set")
        plt.savefig(os.path.join(os.getcwd(), "feature_ablation.png"))
        plt.close()

    # Compute and return average precision
    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    return test_scikit_ap(cat_preds, cat_labels)