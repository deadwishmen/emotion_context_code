import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random


def get_arg():
    parser = ArgumentParser()
    parser.add_argument('--path_dataset', default='', type=str, help='Path to dataset folder images')
    parser.add_argument('--save_test', default="", type=str, help='path save image visualizer')


    return parser.parse_args()


def display_images(images_array, text_image, num_images=5, save_path="output.png"):
    random_indices = random.sample(range(len(images_array)), num_images)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_array[idx], cmap=plt.cm.binary)
        print(text_image[idx])

    plt.savefig(save_path, bbox_inches='tight', dpi=300)  # Lưu ảnh với chất lượng cao
    plt.show()





def main(args):
    print(args.path_dataset)
    path_csv = os.path.join(args.path_dataset, 'train.csv')
    path_images = os.path.join(args.path_dataset, 'train_context_bbox_arr.npy')
    save_path = os.path.join(args.save_test, 'output.png')
    data_csv = pd.read_csv(path_csv)
    images = np.load(path_images)
    display_images(images, data_csv['Output'].values, save_path = save_path)
    

if __name__ == '__main__':
    arges = get_arg()
    main(arges)
    