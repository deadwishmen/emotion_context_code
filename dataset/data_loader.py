import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
from tokenizer_dataset import tokenizer_dataset

# Function to set normalization and transformations
def set_normalization_and_transforms(isSwinT=False):
    # Normalization values
    context_mean = [0.4690646, 0.4407227, 0.40508908]
    context_std = [0.2514227, 0.24312855, 0.24266963]

    body_mean = [0.43832874, 0.3964344, 0.3706214]
    body_std = [0.24784276, 0.23621225, 0.2323653]

    if isSwinT:
        body_mean = [0.485, 0.456, 0.406]
        body_std = [0.229, 0.224, 0.225]

    face_mean = [0.507395516207, 0.507395516207, 0.507395516207]
    face_std = [0.255128989415, 0.255128989415, 0.255128989415]

    context_norm = [context_mean, context_std]
    body_norm = [body_mean, body_std]
    face_norm = [face_mean, face_std]

    # Transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    face_train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor()
    ])

    face_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    if isSwinT:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=[224]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=[232], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=[224]),
            transforms.ToTensor()
        ])

    return context_norm, body_norm, face_norm, train_transform, test_transform, face_train_transform, face_test_transform

# Emotic_PreDataset class
class Emotic_PreDataset(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''
    def __init__(self, x_context, x_body, x_face, y_cat, y_cont, context_transform, body_transform, face_transform, context_norm, body_norm, face_norm):
        super(Emotic_PreDataset,self).__init__()
        self.x_context = x_context
        self.x_body = x_body
        self.x_face = x_face
        self.y_cat = y_cat
        self.y_cont = y_cont
        self.context_transform = context_transform
        self.body_transform = body_transform
        self.face_transform = face_transform
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])
        self.face_norm = transforms.Normalize(face_norm[0], face_norm[1])

    def __len__(self):
        return len(self.y_cont)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        image_face = self.x_face[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]
        return self.context_norm(self.context_transform(image_context)), self.body_norm(self.body_transform(image_body)), self.face_norm(self.face_transform(image_face)), torch.tensor(cat_label, dtype=torch.float32), torch.tensor(cont_label, dtype=torch.float32)/10.0

# Load data function
def load_data(data_src, batch_size, train_transform, test_transform, face_train_transform, face_test_transform, context_norm, body_norm, face_norm):
    # Load train data
    train_context = np.load(os.path.join(data_src, 'train_context_arr.npy'))
    train_body = np.load(os.path.join(data_src,'train_body_arr.npy'))
    train_cat = np.load(os.path.join(data_src,'train_cat_arr.npy'))
    train_cont = np.load(os.path.join(data_src,'train_cont_arr.npy'))

    # Load validation data
    val_context = np.load(os.path.join(data_src,'val_context_arr.npy'))
    val_body = np.load(os.path.join(data_src,'val_body_arr.npy'))
    val_cat = np.load(os.path.join(data_src,'val_cat_arr.npy'))
    val_cont = np.load(os.path.join(data_src,'val_cont_arr.npy'))

    # Load test data
    test_context = np.load(os.path.join(data_src,'test_context_arr.npy'))
    test_body = np.load(os.path.join(data_src,'test_body_arr.npy'))
    test_cat = np.load(os.path.join(data_src,'test_cat_arr.npy'))
    test_cont = np.load(os.path.join(data_src,'test_cont_arr.npy'))

    # Load face data
    train_face = np.stack((np.load(os.path.join(data_src,'train_face_arr.npy')),) * 3, axis=-1)
    val_face = np.stack((np.load(os.path.join(data_src,'val_face_arr.npy')),) * 3, axis=-1)
    test_face = np.stack((np.load(os.path.join(data_src,'test_face_arr.npy')),) * 3, axis=-1)

    # Load text data

    train_text = tokenizer_dataset(os.path.join(data_src, 'train.csv'))
    val_text = tokenizer_dataset(os.path.join(data_src, 'val.csv'))
    test_text = tokenizer_dataset(os.path.join(data_src, 'test.csv'))

    

    # Categorical emotion classes
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
           'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
           'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

    print ('train ', 'context ', train_context.shape, 'body', train_body.shape, 'cat ', train_cat.shape, 'cont', train_cont.shape)
    print ('val ', 'context ', val_context.shape, 'body', val_body.shape, 'cat ', val_cat.shape, 'cont', val_cont.shape)
    print ('test ', 'context ', test_context.shape, 'body', test_body.shape, 'cat ', test_cat.shape, 'cont', test_cont.shape)


    cat2ind = {emotion: idx for idx, emotion in enumerate(cat)}
    ind2cat = {idx: emotion for idx, emotion in enumerate(cat)}

    train_dataset = Emotic_PreDataset(train_context, train_body, train_face, train_cat, train_cont, train_transform, train_transform, face_train_transform, context_norm, body_norm, face_norm)
    val_dataset = Emotic_PreDataset(val_context, val_body, val_face, val_cat, val_cont, train_transform, train_transform, face_train_transform, context_norm, body_norm, face_norm)
    test_dataset = Emotic_PreDataset(test_context, test_body, test_face, test_cat, test_cont, test_transform, test_transform, face_test_transform, context_norm, body_norm, face_norm)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, cat2ind, ind2cat, train_dataset.__len__(), val_dataset.__len__(), test_dataset.__len__()
