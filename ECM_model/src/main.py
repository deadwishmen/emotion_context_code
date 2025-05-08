import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import initialize_models
from dataset import get_data_loaders
from train import train
from test import test_disc, test_cont
import os
import requests
import tarfile
import subprocess

def download_places365_weights(store_path="./places", file_path="./places/resnet50_places365.pth.tar"):
    """
    Kiểm tra và tải trọng số ResNet-50 Places365 nếu chưa tồn tại.
    Xử lý cả trường hợp tệp là .tar hoặc .pth trực tiếp.
    Trả về đường dẫn đến tệp .pth.
    """
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    
    # Tệp đích cuối cùng là .pth
    pth_file = file_path.replace(".pth.tar", ".pth")
    
    # Nếu tệp .pth đã tồn tại, trả về ngay
    if os.path.exists(pth_file):
        return pth_file
    
    # Tải tệp nếu chưa tồn tại
    if not os.path.exists(file_path):
        print(f"Downloading ResNet-50 Places365 weights to {file_path}")
        url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            else:
                raise Exception(f"Failed to download weights from {url}. Status code: {response.status_code}")
        except Exception as e:
            raise Exception(f"Error downloading weights: {str(e)}")
    
    # Kiểm tra xem tệp có phải là .tar và giải nén
    try:
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(store_path)
            print(f"Extracted {file_path} to {store_path}")
    except tarfile.ReadError:
        print(f"File {file_path} is not a valid tar archive. Assuming it is a .pth file.")
        # Nếu không phải .tar, giả sử là .pth và đổi tên
        os.rename(file_path, pth_file)
    except Exception as e:
        raise Exception(f"Error extracting {file_path}: {str(e)}")
    
    # Kiểm tra xem tệp .pth đã được tạo chưa
    if not os.path.exists(pth_file):
        raise FileNotFoundError(f"Could not find {pth_file} after extraction")
    
    return pth_file

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Branch Network for Imagery Emotion Prediction")
    parser.add_argument('--context-backbone', type=str, default='resnet50_places365',
                        choices=['resnet50_places365', 'resnet18_imagenet', 'resnet50_imagenet'],
                        help='Backbone for context branch')
    parser.add_argument('--context-weights', type=str, default='./places/resnet50_places365.pth',
                        help='Path to context backbone weights')
    parser.add_argument('--body-backbone', type=str, default='swint_emotic',
                        choices=['resnet18_imagenet', 'resnet50_imagenet', 'resnet50_emotic', 'swint_imagenet', 'swint_emotic'],
                        help='Backbone for body branch')
    parser.add_argument('--body-weights', type=str, default='/path/to/emotic/swint.pth',
                        help='Path to body backbone weights')
    parser.add_argument('--face-backbone', type=str, default='bfer',
                        choices=['resnet18_imagenet', 'bfer'],
                        help='Backbone for face branch')
    parser.add_argument('--face-weights', type=str, default='/path/to/bfer/model.pth',
                        help='Path to face backbone weights')
    parser.add_argument('--vad-prediction', action='store_true',
                        help='Perform continuous (VAD) prediction instead of discrete')
    parser.add_argument('--data-dir', type=str, default='/path/to/emotic/data',
                        help='Directory containing EMOTIC dataset images')
    parser.add_argument('--train-mat', type=str, default='/path/to/train.mat',
                        help='Path to training .mat file')
    parser.add_argument('--val-mat', type=str, default='/path/to/val.mat',
                        help='Path to validation .mat file')
    parser.add_argument('--test-mat', type=str, default='/path/to/test.mat',
                        help='Path to testing .mat file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--model-path', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'], help='Device to run the model on')
    return parser.parse_args()

def main():
    args = parse_args()

    # Tự động tải trọng số Places365 nếu cần
    if args.context_backbone == "resnet50_places365" and not os.path.exists(args.context_weights):
        args.context_weights = download_places365_weights()

    # Configuration dictionary based on parsed arguments
    config = {
        'context_backbone': args.context_backbone,
        'context_weights': args.context_weights,
        'body_backbone': args.body_backbone,
        'body_weights': args.body_weights,
        'face_backbone': args.face_backbone,
        'face_weights': args.face_weights,
        'isVADPrediction': args.vad_prediction,
        'data_dir': args.data_dir,
        'train_mat': args.train_mat,
        'val_mat': args.val_mat,
        'test_mat': args.test_mat,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'model_path': args.model_path,
        'lr': args.lr,
        'device': torch.device(args.device)
    }

    # Initialize models
    model_context, model_body, model_face, fusion_model = initialize_models(config)
    models = [model_context, model_body, model_face, fusion_model]

    # Data loaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_data_loaders(
        config['train_mat'], config['val_mat'], config['test_mat'], config['data_dir'], config['batch_size']
    )

    # Optimizer and scheduler
    opt = optim.Adam(list(fusion_model.parameters()) + list(model_context.parameters()) +
                     list(model_body.parameters()) + list(model_face.parameters()), lr=config['lr'])
    scheduler = StepLR(opt, step_size=5, gamma=0.1)

    # Loss functions
    disc_loss = nn.BCEWithLogitsLoss()
    cont_loss = nn.MSELoss()

    # Training
    train(config['epochs'], config['model_path'], opt, scheduler, models, disc_loss, cont_loss,
          config['device'], train_loader, val_loader, train_dataset, val_dataset, config['isVADPrediction'])

    # Testing
    model_context = torch.load(os.path.join(config['model_path'], 'model_context.pth'))
    model_body = torch.load(os.path.join(config['model_path'], 'model_body.pth'))
    model_face = torch.load(os.path.join(config['model_path'], 'model_face.pth'))
    fusion_model = torch.load(os.path.join(config['model_path'], 'model_fusion.pth'))
    models = [model_context, model_body, model_face, fusion_model]

    if config['isVADPrediction']:
        test_mae = test_cont(models, config['device'], test_loader, len(test_dataset))
        print(f'testing MAE={test_mae:.4f}')
    else:
        test_map = test_disc(models, config['device'], test_loader, len(test_dataset))
        print(f'testing mAP={test_map:.4f}')

if __name__ == '__main__':
    main()