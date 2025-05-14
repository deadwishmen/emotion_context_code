import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import test_scikit_ap, get_thresholds
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt
import os


def test_disc(models, device, data_loader, num_images, combine="default", choices_model_context=None, xai=False):
    """
    Test the model ensemble on a dataset and optionally perform feature ablation.

    Args:
        models: Tuple of (model_context, model_body, model_face, model_text, fusion_model)
        device: Torch device (e.g., 'cuda' or 'cpu')
        data_loader: DataLoader for the test dataset
        num_images: Total number of images in the dataset
        combine: Combination method ('q_former' or 'default')
        choices_model_context: Context model type ('vit' or other)
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
        for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc="Testing")):
            images_context, images_body, images_face, tokenizer_text, labels_cat, labels_cont = batch

            # Move inputs to device
            images_context = images_context.to(device)
            images_body = images_body.to(device)
            images_face = torch.mean(images_face, dim=1, keepdim=True).to(device)
            tokenizer_text = {k: v.to(device) for k, v in tokenizer_text.items()}
            labels_cat = labels_cat.to(device)
            labels_cont = labels_cont.to(device)

            # Get current batch size
            current_batch_size = images_context.size(0)

            # Forward pass
            pred_body = model_body(images_body)
            pred_face = model_face(images_face)
            pred_text = model_text(**tokenizer_text).last_hidden_state
            if choices_model_context == "vit":
                pred_context = model_context.get_image_features(pixel_values=images_context)
            else:
                pred_context = model_context(images_context)

            # Debug: Check shapes of inputs to fusion_model
            print(f"Batch {batch_idx}:")
            print(f"  images_context shape: {images_context.shape}")
            print(f"  pred_context shape: {pred_context.shape}")
            print(f"  pred_body shape: {pred_body.shape}")
            print(f"  pred_face shape: {pred_face.shape}")
            print(f"  pred_text shape: {pred_text.shape}")
            print(f"  labels_cat shape: {labels_cat.shape}")

            # Process pred_text and pred_context based on combine
            if combine == "q_former":
                pred_text = pred_text
                pred_context = pred_context[:, 1:]
            else:
                pred_text = pred_text.mean(dim=1)

            # Debug: Check shapes after processing
            print(f"  pred_text (after processing) shape: {pred_text.shape}")
            print(f"  pred_context (after processing) shape: {pred_context.shape}")

            # Call fusion_model
            try:
                pred_cat = fusion_model(pred_context, pred_body, pred_face, pred_text)
            except Exception as e:
                print(f"Error in fusion_model: {e}")
                raise

            # Debug: Check pred_cat shape
            print(f"  pred_cat shape: {pred_cat.shape}, expected: ({current_batch_size}, 26)")

            # Ensure pred_cat has correct shape
            if pred_cat.shape != (current_batch_size, 26):
                print(f"Warning: pred_cat shape {pred_cat.shape} does not match expected ({current_batch_size}, 26)")
                if pred_cat.shape == (26, 26):
                    print("Attempting temporary fix by selecting first batch_size rows")
                    pred_cat = pred_cat[:current_batch_size, :]  # Temporary fix
                else:
                    raise ValueError(f"Unexpected pred_cat shape: {pred_cat.shape}, expected: ({current_batch_size}, 26)")

            # Feature ablation if enabled
            if xai:
                try:
                    attr = ablation.attribute((pred_context, pred_body, pred_face, pred_text), target=0)
                    attr_numpy = [a.cpu().numpy().mean() for a in attr]
                    all_importance.append(attr_numpy)
                except Exception as e:
                    print(f"Error in feature ablation: {e}")

            # Store predictions and labels
            cat_preds[idx:idx + current_batch_size] = pred_cat.cpu().numpy()
            cat_labels[idx:idx + current_batch_size] = labels_cat.cpu().numpy()
            idx += current_batch_size

    # Save thresholds
    thresholds = get_thresholds(cat_preds, cat_labels)
    np.save(os.path.join(os.getcwd(), 'thresholds.npy'), thresholds)

    # Visualize feature importance if XAI is enabled
    if xai and all_importance:
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
    test_ap = test_scikit_ap(cat_preds, cat_labels)
    print(f"Test AP: {test_ap:.4f}")
    return test_ap