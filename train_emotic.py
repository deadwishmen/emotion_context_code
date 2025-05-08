import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from argparse import ArgumentParser, ArgumentTypeError
from transformers import AutoModel
from model.resnet import resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b
from model.vit import vit_b_16
from model.fusion import FusionModel, FusionConcatModel, TransformerFusionModel, DualPathAttentionFusion, QFormer
from dataset.data_loader import load_data, set_normalization_and_transforms
from utils.losses import DiscreteLoss, BCEWithLogitsLoss, FocalLoss
from training.training import train_disc
import matplotlib.pyplot as plt
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="Train a multi-modal fusion model for emotion prediction")
    parser.add_argument('--save_model', default='./save_model', type=str, help="Path to save trained models")
    parser.add_argument('--batch_size', default=26, type=int, help="Batch size for training")
    parser.add_argument('--epochs', default=25, type=int, help="Number of training epochs")
    parser.add_argument('--loss', default='L2', type=str, choices=['L2', 'BCE', 'CrossEntropy', 'FocalLoss'], help="Loss function")
    parser.add_argument('--model_body', default='swin-t', type=str, choices=['swin-t', 'swin-s', 'swin-b', 'vit'], help="Body model")
    parser.add_argument('--model_context', default='resnet', type=str, choices=['resnet', 'vit'], help="Context model")
    parser.add_argument('--model_text', default='distilbert', type=str, choices=['distilbert', 'bert', 'roberta', 'deberta'], help="Text model")
    parser.add_argument('--path_dataset', default='/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full', type=str, help="Path to dataset")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay")
    parser.add_argument('--step_size', default=7, type=int, help="Step size for LR scheduler")
    parser.add_argument('--gamma', default=0.1, type=float, help="Gamma for LR scheduler")
    parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat', 'sum', 'avg', 'transformer', 'attention', 'q_former'], help="Fusion method")
    return parser.parse_args()


def initialize_models(args):
    """Initialize context, body, face, text, and fusion models based on arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_context_features = 768

    # Context model
    if args.fusion_method == "q_former":
        model_context = vit_b_16(pretrained=True)
    elif args.model_context == "resnet":
        model_context = resnet50_place365(pretrained=True)
        num_context_features = list(model_context.children())[-1].in_features
    else:
        model_context = vit_b_16(pretrained=True)

    # Body model
    model_dict = {
        "swin-t": swin_v2_t,
        "swin-s": swin_v2_s,
        "swin-b": swin_v2_b,
        "vit": vit_b_16
    }
    model_body = model_dict.get(args.model_body, vit_b_16)(pretrained=True)

    # Face model
    model_face = cnn_face(pretrained=True)

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
    if isinstance(last_layer, nn.Sequential):
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

    # Freeze pre-trained models, only train fusion and text models
    for model in [model_context, model_body, model_face]:
        for param in model.parameters():
            param.requires_grad = False
    for param in fusion_model.parameters():
        param.requires_grad = True
    for param in model_text.parameters():
        param.requires_grad = True

    return model_context, model_body, model_face, model_text, fusion_model, device


def train(args):
    """Train the multi-modal fusion model."""
    # Initialize data
    context_norm, body_norm, face_norm, train_transform, test_transform, face_train_transform, face_test_transform = set_normalization_and_transforms(args.model_body)
    train_loader, val_loader, _, _, _, train_length, val_length, _ = load_data(
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

    # Initialize models
    model_context, model_body, model_face, model_text, fusion_model, device = initialize_models(args)

    # Optimizer and scheduler
    params = list(fusion_model.parameters()) + list(model_text.parameters())
    optimizer = optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Loss function
    loss_map = {
        "L2": DiscreteLoss('dynamic', device),
        "BCE": BCEWithLogitsLoss('dynamic', device),
        "FocalLoss": FocalLoss(gamma=2.0, alpha=None, weight_type='mean', device=device)
    }
    disc_loss = loss_map[args.loss]

    # Train model
    train_loss, val_loss, train_mae, val_mae = train_disc(
        args.epochs,
        args.save_model,
        optimizer,
        scheduler,
        [model_context, model_body, model_face, model_text, fusion_model],
        disc_loss,
        cat_loss_param=1.0,
        cont_loss_param=0.0,
        train_length=train_length,
        val_length=val_length,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        conbine=args.fusion_method,
        choices_model_context=args.model_context
    )

    # Plot training results
    f, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(15, 10))
    f.suptitle('Multi-Branch Network for Imagery Emotion Prediction')
    ax1.plot(range(len(train_loss)), train_loss, color='Blue')
    ax2.plot(range(len(val_loss)), val_loss, color='Red')
    ax1.legend(['train loss'])
    ax2.legend(['val loss'])
    ax3.plot(range(len(train_mae)), train_mae, color='Blue')
    ax4.plot(range(len(val_mae)), val_mae, color='Red')
    ax3.legend(['train mAP'])
    ax4.legend(['val mAP'])
    plt.savefig(os.path.join(args.save_model, 'training_plots.png'))
    plt.close()


if __name__ == '__main__':
    args = get_args()
    train(args)