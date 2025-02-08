import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def get_arg():
    parser = ArgumentParser()
    parser.add_argument('--path_npy', default='', type=str, help='Path to dataset file images npy')
    parser.add_argument('--path_csv', default='', type=str, help='Path to csv file')

    return parser.parse_args()


def display_images(images_array, text_image, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images_array[i], cmap=plt.cm.binary)
        plt.xlabel(text_image[i])
    plt.show()




def main(args):
    print(args.path_dataset)
    path_csv = os.path.join(args.path_dataset, 'train.csv')
    path_images = os.path.join(args.path_dataset, 'train_images.npy')

    data_csv = pd.read_csv(path_csv)
    images = np.load(path_images)
    display_images(images, data_csv['Output'].values)
    print(data_csv.head())

    
    # display_images(images, labels)

if __name__ == '__main__':
    arges = get_arg()
    main(arges)
    