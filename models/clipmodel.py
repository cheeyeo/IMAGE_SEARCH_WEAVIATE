from typing import List
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import numpy as np


class ClipImageEmbed:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    
    def embed(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        self.model.eval()

        with torch.no_grad():
            inputs = self.processor(text=None, images=image, return_tensors='pt', padding=True).to(self.device)

            image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            return image_embeds.to('cpu').numpy()[0]
    

if __name__ == '__main__':
    model = ClipImageEmbed()
    res = model.embed('testimgs/test.jpg')
    print(type(res))
    print(res)
    print(res.shape)