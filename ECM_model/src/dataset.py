import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy.io
import os
from torchvision import transforms

class EmoticDataset(Dataset):
    def __init__(self, mat_file, data_dir, transform_context=None, transform_body=None, transform_face=None):
        self.data_dir = data_dir
        self.transform_context = transform_context
        self.transform_body = transform_body
        self.transform_face = transform_face
        self.data = scipy.io.loadmat(mat_file)['data']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        context_path = os.path.join(self.data_dir, sample[0][0])
        body_path = os.path.join(self.data_dir, sample[1][0])
        face_path = os.path.join(self.data_dir, sample[2][0])
        
        context_img = Image.open(context_path).convert('RGB')
        body_img = Image.open(body_path).convert('RGB')
        face_img = Image.open(face_path).convert('RGB')

        labels_cat = torch.tensor(sample[3], dtype=torch.float32)
        labels_cont = torch.tensor(sample[4], dtype=torch.float32)

        if self.transform_context:
            context_img = self.transform_context(context_img)
        if self.transform_body:
            body_img = self.transform_body(body_img)
        if self.transform_face:
            face_img = self.transform_face(face_img)

        return context_img, body_img, face_img, labels_cat, labels_cont

def get_transforms():
    transform_context = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_body = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_face = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_context, transform_body, transform_face

def get_data_loaders(train_mat, val_mat, test_mat, data_dir, batch_size=32):
    transform_context, transform_body, transform_face = get_transforms()
    
    train_dataset = EmoticDataset(train_mat, data_dir, transform_context, transform_body, transform_face)
    val_dataset = EmoticDataset(val_mat, data_dir, transform_context, transform_body, transform_face)
    test_dataset = EmoticDataset(test_mat, data_dir, transform_context, transform_body, transform_face)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset