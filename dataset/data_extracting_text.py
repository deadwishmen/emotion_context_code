import torch
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn.modules import activation
from transformers import BitsAndBytesConfig
from transformers import pipeline
from argparse import ArgumentParser



row = [ 'Index', 'Folder', 'Filename',
        'Image Size', 'BBox', 'Categorical_Labels',
        'Continuous_Labels', 'Gender', 'Age']

cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
        'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
        'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

prompts = "USER: <image>\ Given the following list of emotions: suffering, pain, aversion, disapproval, anger, fear, annoyance, fatigue, disquietment, doubt/confusion, embarrassment, disconnection, affection, confidence, engagement, happiness, peace, pleasure, esteem, excitement, anticipation, yearning, sensitivity, surprise, sadness, and sympathy. Based on the image context, please choose which emotions are more suitable for describing how the person in the red box feels and explain in detail why you choose these emotions according to the aspects: actions and postures of the person in the red box, the context surrounding the person in the red box.\nASSISTANT:"



prompt_v2 = f"USER: <image>\\ Given the following list of emotions: {', '.join(cat)} explain in detail which emotions are more suitable for describing how the person in the red box feels based on the image context\nASSISTANT:"


def get_arg():
  parser = ArgumentParser()
  parser.add_argument('--path_dataset', default= '/content/drive/MyDrive/DatMinhNe/Dataset/emotic_obj_full_v2', type=str)
  parser.add_argument('--path_save', default='/content', type=str)
  parser.add_argument('--model_id', default="llava-hf/llava-1.5-7b-hf", type=str, choices=["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf"])
  parser.add_argument('--bit8', action = 'store_true')
  parser.add_argument('--max_new_tokens', default=250, type=int)

  args = parser.parse_args()
  return args


def demo_shape(image_numpy):
  image_pil = Image.fromarray(image_numpy)
  image_pil.save("/content/extracted_image.png")

def np2pil(image_numpy):
  image_pil = Image.fromarray(image_numpy)
  return image_pil

def np_list2pil_list(image_list):
  image_pil_list = [Image.fromarray(image) for image in image_list]
  return image_pil_list

def get_assistant_text(text):
  start_index = text.find("ASSISTANT")
  assistant_text = text[start_index:] if start_index != -1 else ""
  return assistant_text

def processor_image2text(images, pipe, max_new_tokens):
  outputs = pipe(images, text=prompts, generate_kwargs={"max_new_tokens": max_new_tokens})
  return outputs[0]["generated_text"]


def data_extracting(args):
  model_id = args.model_id
  bit8 = args.bit8
  max_new_tokens = args.max_new_tokens
  path_dataset = args.path_dataset
  path_save = args.path_save
  path_dataset_train = os.path.join(path_dataset, 'train.csv')
  path_dataset_val = os.path.join(path_dataset, 'val.csv')
  path_dataset_test = os.path.join(path_dataset, 'test.csv')

  path_save_train = os.path.join(path_save, 'train.csv')
  path_save_val = os.path.join(path_save, 'val.csv')
  path_save_test = os.path.join(path_save, 'test.csv')

  train_context_bbox_arr = os.path.join(path_dataset, 'train_context_bbox_arr.npy')
  val_context_bbox_arr = os.path.join(path_dataset, 'val_context_bbox_arr.npy')
  test_context_bbox_arr = os.path.join(path_dataset, 'test_context_bbox_arr.npy')

  # train_context_bbox_arr = np.load(train_context_bbox_arr)
  # image_list_train = [train_context_bbox_arr[i] for i in range(train_context_bbox_arr.shape[0])]
  # del train_context_bbox_arr
  # image_list_train = np_list2pil_list(image_list_train)
  # val_context_bbox_arr = np.load(val_context_bbox_arr)
  # image_list_val = [val_context_bbox_arr[i] for i in range(val_context_bbox_arr.shape[0])]
  # del val_context_bbox_arr
  # image_list_val = np_list2pil_list(image_list_val)
  test_context_bbox_arr = np.load(test_context_bbox_arr)
  image_list_test = [test_context_bbox_arr[i] for i in range(test_context_bbox_arr.shape[0])]
  del test_context_bbox_arr
  image_list_test = np_list2pil_list(image_list_test)




  


  train_csv = pd.read_csv(path_dataset_train)
  val_csv = pd.read_csv(path_dataset_val)
  test_csv = pd.read_csv(path_dataset_test)


  if 'Output' not in test_csv.columns:
    test_csv['Output'] = ""

  if bit8 is False:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
  else:
    quantization_config = BitsAndBytesConfig(
      load_in_8bit=True,
      bnb_8bit_compute_dtype=torch.float16
    )




  pipe = pipeline("image-text-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
  for idx, image in enumerate(tqdm(image_list_test, desc="Processing images")):
    output = processor_image2text(image, pipe, max_new_tokens)
    output = get_assistant_text(output)
    test_csv.loc[idx, 'Output'] = output
    test_csv.to_csv(path_dataset_test, index=False)




if __name__=='__main__':
  args = get_arg()
  data_extracting(args)
