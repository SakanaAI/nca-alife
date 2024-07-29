from transformers import AutoProcessor, CLIPModel
import torch

from einops import rearrange

class MyTorchCLIP():
    def __init__(self, clip_model, device=None, dtype=None):
        self.processor = AutoProcessor.from_pretrained(f"openai/{clip_model}")
        self.clip_model = CLIPModel.from_pretrained(f"openai/{clip_model}").to(device, dtype)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.img_mean = torch.tensor(self.processor.image_processor.image_mean, device=device, dtype=dtype)
        self.img_std = torch.tensor(self.processor.image_processor.image_std, device=device, dtype=dtype)
        self.device, self.dtype = device, dtype

    def embed_text(self, prompts):
        """
        prompts is list of strings
        returns shape (B D)
        """
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            z_text = self.clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return z_text/z_text.norm(dim=-1, keepdim=True)
    
    def embed_img(self, img):
        """
        img shape (B C H W) and values in [0, 1].
        returns shape (B D)
        """
        img = (img-self.img_mean[:, None, None])/self.img_std[:, None, None]
        z_img = self.clip_model.get_image_features(img)
        return z_img/z_img.norm(dim=-1, keepdim=True)