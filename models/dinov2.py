import torch
import torchvision.transforms as T
from PIL import Image


class DinoV2Embed:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.to(self.device)

        self.preprocessor = T.Compose([
            T.Resize((244, 244)),
            T.CenterCrop(224),
            # T.Normalize([0.5], [0.5])
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    
    def embed(self, image_path):
        img = Image.open(image_path).convert('RGB')
        self.model.eval()

        with torch.no_grad():
            img = self.preprocessor(img)[:3].unsqueeze(0).to(self.device)

            embedding = self.model(img)
            embedding2 = embedding[0].cpu().numpy()

            return embedding2
        

if __name__ == "__main__":
    model = DinoV2Embed()
    res = model.embed('testimgs/test.jpg')
    print(res.shape)
    print(type(res))
    print(res)