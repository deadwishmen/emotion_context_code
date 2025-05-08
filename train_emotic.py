import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
from tqdm import tqdm
from model.resnet import resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b
from model.vit import vit_b_16
from model.fusion import FusionModel, FusionConcatModel, TransformerFusionModel, DualPathAttentionFusion, QFormer
from transformers import AutoModel
from utils.metrics import test_scikit_ap, get_thresholds
from dataset.data_loader import load_data, set_normalization_and_transforms
from captum.attr import FeatureAblation
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description="Test multi-modal fusion model for emotion prediction")
    parser.add_argument('--save_model', default='./save_model', type=str, help="Path to saved models")
    parser.add_argument('--batch_size', default=26, type=int, help="Batch size for testing")
    parser.add_argument('--path_dataset', default='/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full', type=str, help="Path to dataset")
    parser.add_argument('--model_body', default='swin-t', type=str, choices=['swin-t', 'swin-s', 'swin-b', 'vit'], help="Body model")
    parser.add_argument('--model_context', default='resnet', type=str, choices=['resnet', 'vit'], help="Context model")
    parser.add_argument('--model_text', default='distilbert', type=str, choices=['distilbert', 'bert', 'roberta', 'deberta'], help="Text model")
    parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat', 'sum', 'avg', 'transformer', 'attention', 'q_former'], help="Fusion method")
    parser.add_argument('--xai', type=str2bool, default=False, help="Enable feature ablation")
    return parser.parse_args()


def load_models(args):
    """Load pre-trained models and initialize fusion model."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_context_features = 768

    # Context model
    if args.fusion_method == "q_former":
        model_context = vit_b_16(pretrained=False)
    elif args.model_context == "resnet":
        model_context = resnet50_place365(pretrained=False)
        num_context_features = list(model_context.children())[-1].in_features
    else:
        model_context = vit_b_16(pretrained=False)

    # Body model
    model_dict = {
        "swin-t": swin_v2_t,
        "swin-s": swin_v2_s,
        "swin-b": swin_v2_b,
        "vit": vit_b_16
    }
    model_body = model_dict.get(args.model_body, vit_b_16)(pretrained=False)

    # Face model
    model_face = cnn_face(pretrained=False)

    # Text model
    text_model_map = {
        "distilbert": 'distilbert-base-uncased',
        "bert": 'bert-base-uncased',
        "roberta": 'roberta-base',
        "deberta": 'microsoft/deberta-v2-xlarge'
    }
    model_text = AutoModel.from_pretrained(text_model_map[args.model_text])

    # Get feature dimensions
    last_layer = list(model_body.children())[-1]
    if isinstance(last_layer, torch.nn.Sequential):
        last_layer = list(last_layer.children())[-1]
    if not hasattr(last_layer, 'in_features'):
        raise ValueError("The last layer of body model has no in_features.")
    num_body_features = last_layer.in_features
    num_face_features = list(model_face.children())[-3].in_features
    num_text_features = model_text.config.hidden_size

    # Fusion model
    fusion_model_map = {
        "concat": FusionConcatModel,
        "sum": FusionModel,
        "avg": FusionModel,
        "transformer": TransformerFusionModel,
        "attention": DualPathAttentionFusion,
        "q_former": QFormer
    }
    fusion_class = fusion_model_map[args.fusion_method]
    if args.fusion_method in ["sum", "avg"]:
        fusion_model = fusion_class(num_context_features, num_body_features, num_face_features, num_text_features, args.fusion_method)
    else:
        fusion_model = fusion_class(num_context_features, num_body_features, num_face_features, num_text_features)

    # Load saved weights
    model_context.load_state_dict(torch.load(os.path.join(args.save_model, "model_context.pth"), map_location=device))
    model_body.load_state_dict(torch.load(os.path.join(args.save_model, "model_body.pth"), map_location=device))
    model_face.load_state_dict(torch.load(os.path.join(args.save_model, "model_face.pth"), map_location=device))
    model_text.load_state_dict(torch.load(os.path.join(args.save_model, "model_text.pth"), map_location=device))
    fusion_model.load_state_dict(torch.load(os.path.join(args.save_model, "model_fusion.pth"), map_location=device))

    return [model_context, model_body, model_face, model_text, fusion_model], device


def test_disc(models, device, data_loader, num_images, args, combine="default", xai=False):
    """
    Test the model ensemble on a dataset and optionally perform feature ablation.

    Args:
        models: Tuple of (model_context, model_body, model_face, model_text, fusion_model)
        device: Torch device (e.g., 'cuda' or 'cpu')
        data_loader: DataLoader for the test dataset
        num_images: Total number of images in the dataset
        args: ArgumentParser object containing save_model path
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
            elif args.model_context == "vit":
                pred_context = model_context.get_image_features(pixel_values=images_context)
                pred_text = pred_text.mean(dim=1)
            else:
                pred_text = pred_text.mean(dim=1)
                pred_context = pred_context

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
    np.save(os.path.join(args.save_model, 'thresholds.npy'), thresholds)

    # Visualize feature importance if XAI is enabled
    if xai:
        all_importance = np.array(all_importance).mean(axis=0)
        feature_groups = ["Context", "Body", "Face", "Text"]
        plt.figure(figsize=(8, 5))
        plt.bar(feature_groups, all_importance, color=["blue", "orange", "green", "red"])
        plt.xlabel("Feature Groups")
        plt.ylabel("Importance")
        plt.title("Feature Ablation on Test Set")
        plt.savefig(os.path.join(args.save_model, "feature_ablation.png"))
        plt.close()

    # Compute and return average precision
    cat_preds = cat_preds.transpose()
    cat_labels = cat_labels.transpose()
    return test_scikit_ap(cat_preds, cat_labels)


def visualize_inputs_and_predictions(
    images_context, images_body, images_face, tokenizer_text, 
    cat_preds, cat_labels, batch_idx, args, tokenizer=None
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
        args: ArgumentParser object containing save_model path
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
    plt.savefig(os.path.join(args.save_model, f"input_visualization_batch_{batch_idx}.png"))
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
    plt.savefig(os.path.join(args.save_model, f"prediction_visualization_batch_{batch_idx}.png"))
    plt.close()


def main():
    args = get_args()
    models, device = load_models(args)

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

    # Run test
    test_map = test_disc(models, device, test_loader, test_length, args, combine=args.fusion_method, xai=args.xai)
    print(f"Testing mAP: {test_map:.4f}")

    # Optional: Visualize inputs and predictions for the first batch
    if args.xai:
        for idx, batch in enumerate(test_loader):
            if idx >= 1:  # Visualize only the first batch
                break
            images_context, images_body, images_face, tokenizer_text, labels_cat, _ = batch
            with torch.no_grad():
                images_context = images_context.to(device)
                images_body = images_body.to(device)
                images_face = images_face.to(device).mean(dim=1, keepdim=True)
                tokenizer_text = {k: v.to(device) for k, v in tokenizer_text.items()}
                pred_body = models[1](images_body)
                pred_face = models[2](images_face)
                pred_text = models[3](**tokenizer_text).last_hidden_state
                pred_context = models[0](images_context)
                if args.fusion_method == "q_former":
                    pred_text = pred_text
                    pred_context = pred_context[:, 1:]
                elif args.model_context == "vit":
                    pred_context = models[0].get_image_features(pixel_values=images_context)
                    pred_text = pred_text.mean(dim=1)
                else:
                    pred_text = pred_text.mean(dim=1)
                    pred_context = pred_context
                pred_cat = models[4](pred_context, pred_body, pred_face | pred_text)
            visualize_inputs_and_predictions(
                images_context, images_body, images_face, tokenizer_text,
                pred_cat.cpu().numpy(), labels_cat.numpy(), idx, args
            )


if __name__ == '__main__':
    main()