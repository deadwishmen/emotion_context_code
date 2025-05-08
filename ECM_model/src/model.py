import torch
import torch.nn as nn
import torchvision.models as models

# Global variables for backbone selection
isSwinT = False
isBFER = False
attention = False
num_context_features = 0
num_body_features = 0
num_face_features = 0

def load_body_backbone(backbone_type, weights_path=None):
    global num_body_features, isSwinT
    if backbone_type == "resnet18_imagenet":
        model_body = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_body_features = list(model_body.children())[-1].in_features
    elif backbone_type == "resnet50_imagenet":
        model_body = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_body_features = list(model_body.children())[-1].in_features
    elif backbone_type == "resnet50_emotic":
        model_body = torch.load(weights_path)
        num_body_features = list(model_body.children())[-1].in_features
    elif backbone_type == "swint_imagenet":
        isSwinT = True
        model_body = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        num_body_features = model_body.head.in_features
    elif backbone_type == "swint_emotic":
        isSwinT = True
        model_body = torch.load(weights_path)
        num_body_features = model_body.head.in_features
    else:
        raise ValueError("Unsupported body backbone")
    return model_body

def load_context_backbone(backbone_type, weights_path=None):
    global num_context_features
    if backbone_type == "resnet50_places365":
        model_context = models.resnet50(weights=None)
        if weights_path:
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            # Loại bỏ tiền tố 'module.' nếu cần
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model_context.load_state_dict(state_dict)
        else:
            raise ValueError("Path to Places365 weights is required for resnet50_places365")
        num_context_features = list(model_context.children())[-1].in_features
    elif backbone_type == "resnet18_imagenet":
        model_context = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_context_features = list(model_context.children())[-1].in_features
    elif backbone_type == "resnet50_imagenet":
        model_context = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_context_features = list(model_context.children())[-1].in_features
    else:
        raise ValueError("Unsupported context backbone")
    return model_context

def load_face_backbone(backbone_type, weights_path=None):
    global num_face_features, isBFER
    if backbone_type == "resnet18_imagenet":
        model_face = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_face_features = list(model_face.children())[-1].in_features
    elif backbone_type == "bfer":
        isBFER = True
        model_face = torch.load(weights_path)
        num_face_features = list(model_face.children())[-1].in_features
    else:
        raise ValueError("Unsupported face backbone")
    return model_face

class FusionNet(nn.Module):
    def __init__(self, num_context_features, num_body_features, num_face_features, num_cat=26, num_cont=3, isVADPrediction=False):
        super(FusionNet, self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.num_face_features = num_face_features
        self.isVADPrediction = isVADPrediction

        self.fc_context = nn.Linear(num_context_features, 512)
        self.fc_body = nn.Linear(num_body_features, 512)
        self.fc_face = nn.Linear(num_face_features, 512)
        self.fc2 = nn.Linear(512 * 3, 512)
        self.fc3 = nn.Linear(512, 256)

        if isVADPrediction:
            self.classifier = nn.Linear(256, num_cont)
        else:
            self.classifier = nn.Linear(256, num_cat)

    def forward(self, x_context, x_body, x_face):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        face_features = x_face.view(-1, self.num_face_features)

        context_features = self.fc_context(context_features)
        body_features = self.fc_body(body_features)
        face_features = self.fc_face(face_features)

        fuse_features = torch.cat((context_features, body_features, face_features), 1)
        fuse_features = self.fc2(fuse_features)
        fuse_features = self.fc3(fuse_features)
        output = self.classifier(fuse_features)
        return output

def initialize_models(config):
    model_context = load_context_backbone(config['context_backbone'], config.get('context_weights'))
    model_body = load_body_backbone(config['body_backbone'], config.get('body_weights'))
    model_face = load_face_backbone(config['face_backbone'], config.get('face_weights'))
    fusion_model = FusionNet(num_context_features, num_body_features, num_face_features, isVADPrediction=config['isVADPrediction'])
    return model_context, model_body, model_face, fusion_model