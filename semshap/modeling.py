import torch.nn as nn
from sentence_transformers import util
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class OFAWrapper(nn.Module):
    def __init__(self, ref_e, model, tokenizer, similarity_model, device='cpu', resolution=(256, 256), logger=None):
        super(OFAWrapper, self).__init__()

        self.ref_e = ref_e
        self.logger = logger
        self.device = device

        self.ss_model = similarity_model
        self.model = model
        self.tokenizer = tokenizer

        self.RESOLUTION = resolution

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize(self.RESOLUTION, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def forward(self, masked_imgs, inputs, **kwargs):

        # wrapper for the value function
        if self.ref_e == None:
            raise ValueError(f"a valid reference embedding has to be defined: ref_e = {self.ref_e}")

        if inputs == None:
            raise ValueError(f"please insert a valid input containing the answer for the model")

        cosine_scores = []
        for masked_img in tqdm(masked_imgs):
            masked_img = self.patch_resize_transform(masked_img).unsqueeze(0).to(self.device)
            out_ids = self.model.generate(inputs.to(self.device), patch_images=masked_img, num_beams=5, no_repeat_ngram_size=3)
            gen = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            gen_e = self.ss_model.encode(gen, convert_to_tensor=True).to(self.device)
            cosine_score = util.cos_sim(self.ref_e, gen_e)
            cosine_scores.append(cosine_score.detach().cpu().item())

            if self.logger:
                self.logger.info({"gen": gen, "cos_sim": cosine_score.item()})

        return np.array(cosine_scores)
