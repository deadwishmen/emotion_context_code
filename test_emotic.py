import torch
import argparse
from model.resnet import resnet50_place365
from model.cnn_face import cnn_face
from model.swin_transformer import swin_v2_t, swin_v2_s, swin_v2_b
from model.vit import vit_b_16
from model.fusion import FusionModel, FusionConcatModel, TransformerFusionModel, DualPathAttentionFusion, QFormer
from transformers import AutoModel
from training.testing import test_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description="Run test for multi-modal fusion model")
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

    # Context model
    if args.fusion_method == "q_former" or args.model_context == "vit":
        model_context = vit_b_16(pretrained=False)
    else:
        model_context = resnet50_place365(pretrained=False)

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
    num_context_features = 768 if args.fusion_method == "q_former" or args.model_context == "vit" else list(model_context.children())[-1].in_features
    last_layer = list(model_body.children())[-1]
    if isinstance(last_layer, torch.nn.Sequential):
        last_layer = list(last_layer.children())[-1]
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
    model_context.load_state_dict(torch.load(f"{args.save_model}/model_context.pth", map_location=device))
    model_body.load_state_dict(torch.load(f"{args.save_model}/model_body.pth", map_location=device))
    model_face.load_state_dict(torch.load(f"{args.save_model}/model_face.pth", map_location=device))
    model_text.load_state_dict(torch.load(f"{args.save_model}/model_text.pth", map_location=device))
    fusion_model.load_state_dict(torch.load(f"{args.save_model}/model_fusion.pth", map_location=device))

    return [model_context, model_body, model_face, model_text, fusion_model], device


def run_test_model():
    """Run the test process for the multi-modal fusion model."""
    args = get_args()
    models, device = load_models(args)
    test_map = test_model(models, device, args, xai=args.xai)
    print(f"Testing mAP: {test_map:.4f}")


if __name__ == '__main__':
    run_test_model()