import os

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype

import numpy as np
import cv2

import gradio as gr
from huggingface_hub import hf_hub_download

from model import IAT


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def dark_inference(img):
    model = IAT()
    checkpoint_file_path = './checkpoint/best_Epoch_lol.pth'
    state_dict = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Load model from {checkpoint_file_path}')

    transform = Compose([
        ToTensor(), 
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ConvertImageDtype(torch.float) 
    ])
    input_img = transform(img)
    print(f'Image shape: {input_img.shape}')

    enhanced_img = model(input_img.unsqueeze(0))
    return enhanced_img[0].permute(1, 2, 0).detach().numpy()


def exposure_inference(img):
    model = IAT()
    checkpoint_file_path = './checkpoint/best_Epoch_exposure.pth'
    state_dict = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Load model from {checkpoint_file_path}')

    transform = Compose([
        ToTensor(), 
        ConvertImageDtype(torch.float) 
    ])
    input_img = transform(img)
    print(f'Image shape: {input_img.shape}')

    enhanced_img = model(input_img.unsqueeze(0))
    return enhanced_img[0].permute(1, 2, 0).detach().numpy()


demo = gr.Blocks()
with demo:
    gr.Markdown(
        """
        # IAT
        Gradio demo for <a href='https://github.com/cuiziteng/Illumination-Adaptive-Transformer' target='_blank'>IAT</a>: To use it, simply upload your image, or click one of the examples to load them. Read more at the links below.
        """
    )

    with gr.Box():
        with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image', type='numpy')
                    with gr.Row():
                        dark_button = gr.Button('Low-light Enhancement')
                    with gr.Row():
                        exposure_button = gr.Button('Exposure Correction')
                with gr.Column():
                    res_image = gr.Image(type='numpy', label='Resutls')
        with gr.Row():
            dark_example_images = gr.Dataset(
                components=[input_image], 
                samples=[['dark_imgs/1.jpg'], ['dark_imgs/2.jpg'], ['dark_imgs/3.jpg']]
            )
        with gr.Row():
            exposure_example_images = gr.Dataset(
                components=[input_image], 
                samples=[['exposure_imgs/1.jpg'], ['exposure_imgs/2.jpg'], ['exposure_imgs/3.jpeg']]
            )

    gr.Markdown(
        """
        <p style='text-align: center'><a href='https://arxiv.org/abs/2205.14871' target='_blank'>You Only Need 90K Parameters to Adapt Light: A Light Weight Transformer for Image Enhancement and Exposure Correction</a> | <a href='https://github.com/cuiziteng/Illumination-Adaptive-Transformer' target='_blank'>Github Repo</a></p>
        """
    )

    dark_button.click(fn=dark_inference, inputs=input_image, outputs=res_image)
    exposure_button.click(fn=exposure_inference, inputs=input_image, outputs=res_image)
    dark_example_images.click(fn=set_example_image, inputs=dark_example_images, outputs=dark_example_images.components)
    exposure_example_images.click(fn=set_example_image, inputs=exposure_example_images, outputs=exposure_example_images.components)

demo.launch(enable_queue=True)