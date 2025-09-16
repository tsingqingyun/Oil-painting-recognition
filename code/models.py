import torch
import open_clip
from PIL import Image
# models.py
import torch
import open_clip
from PIL import Image

class ClipEmbedder:
    def __init__(self, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, pil_imgs):
        # pil_imgs: List[PIL.Image]
        imgs = torch.stack([self.preprocess(im) for im in pil_imgs]).to(self.device)
        feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().float().cpu().numpy()  # 明确返回 float32
