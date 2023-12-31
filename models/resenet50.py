# Using pretrained ResNet-50 model as vectorizer

import numpy as np
import torch
import torchvision.models as models
from torchvision.transforms import transforms as T
from PIL import Image


class ResNet50Vectorizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(weights='IMAGENET1K_V1').to(self.device)
        self.layer_output_size = 2048
        self.extraction_layer = self.model._modules.get('avgpool')
        self.preprocess = T.Compose([
            T.Resize(size=(256, 256), interpolation=T.InterpolationMode.BILINEAR),
            # CENTER CROP
            T.CenterCrop(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])


    def embed(self, image_path: str) -> np.ndarray :
        image = Image.open(image_path).convert('RGB')
        self.model.eval()

        with torch.no_grad():
            img = self.preprocess(image).unsqueeze(0).to(self.device)
            my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)
            
            h = self.extraction_layer.register_forward_hook(copy_data)
            self.model(img)
            h.remove()

            # Below returns a vector 
            return my_embedding.numpy()[0, :, 0, 0]
        

if __name__ == '__main__':
    model = ResNet50Vectorizer()
    res = model.embed('test.jpg')
    print(type(res))
    print(res.shape)
    print(res)