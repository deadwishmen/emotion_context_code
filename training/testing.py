import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import test_scikit_ap, get_thresholds
from dataset.data_loader import load_data, set_normalization_and_transforms
import os


def test_model(models, device, args, xai=False):
    """
    Test the multi-modal fusion model on the test dataset.

    Args:
        models: Tuple of (model_context, model_body, model_face, model_text, fusion_model)
        device: Torch device (e.g., 'cuda' or 'cpu')
        args: Arguments containing batch_size, path_dataset, model_body, model_text, fusion_method
        xai: Whether to perform feature ablation (default: False)

    Returns:
        float: Mean average precision score
    """
    model_context, model_body, model_face, model_text, fusion_model = models

    # Load test data
    context_norm, body_norm, face_norm, train_transform, test_transform, face_train_transform, face_test_transform = (
        set_normalization_and_transforms(args.model_body)
    )
    _, _, test_loader, _, _, _, _, test_length = load_data(
        args.path_dataset,
        args.batch_size,
        train_transform,
        test_transform,
        face_train_transform,
        face_test_transform,
        context_norm,
        body_norm,
        face_norm,
        args.model_text
    )

    # Initialize arrays for predictions and labels
    cat_preds = np.zeros((test_length, 26))
    cat_labels = np.zeros((test_length, 26))

    # Move models to device and set to evaluation mode
    for model in models:
        model.to(device)
        model.eval()

    idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
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

            if args.fusion_method == "q_former":
                pred_text = pred_text
                pred_context = pred_context[:, 1:]
            else:
                pred_text = pred_text.mean(dim=1)

            pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)

            # Store predictions and labels
            batch_size = pred_cat.shape[0]
            cat_preds[idx:idx + batch_size] = pred_cat.cpu().numpy()
            cat_labels[idx:idx + batch_size] = labels_cat.cpu().numpy()
            idx += batch_size

    # Save thresholds
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join(args.save_model, 'thresholds.npy'), thresholds)

    # Compute and return average precision
    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    ap_mean = test_scikit_ap(cat_preds, cat_labels)
    return ap_mean