# VL-SHAP

*Official Repo for the paper: ["Interpreting Vision and Language Generative Models with Semantic Visual Priors"](https://arxiv.org/abs/2304.14986).*
---
Explain VL generative models using  **sentence-based** visual explanations, exploiting the model's **visual semantic priors** and **KernelSHAP**

<img align="center" width="950" height="350" 
src="https://drive.google.com/uc?export=view&id=15kivtqVyD8DeL2ueL9qubOCKnEJXwWuA">

### Overview
 

- **🗃️ Repository:** [github.com/michelecafagna26/vl-shap](https://github.com/michelecafagna26/vl-shap)
- **📜 Paper:** [Interpreting Vision and Language Generative Models with Semantic Visual Priors](https://arxiv.org/abs/2304.14986)
- **🖊️ Contact:** michele.cafagna@um.edu.mt


### Requirements

```txt
python == 3.6.9
pytorch
torchvision
```

### Installation

```bash
pip install git+https://github.com/michelecafagna26/vl-shap.git#egg=semshap
```

### Example: Explain OFA Visual Question Answering Model

Install OFA from the [official repo](https://github.com/OFA-Sys/OFA)
Then run the following code to **extract semantic masks**

```python3
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from transformers import OFATokenizer, OFAModel

from semshap.masking import generate_dff_masks, generate_superpixel_masks
from semshap.plot import  heatmap, barh, plot_masks
from semshap.explainers import BaseExplainer


import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

ckpt_dir = "/path/to/the/model/ofa-models/OFA-large" # change this
device = "cuda" if torch.cuda.is_available() else "cpu"
img_url="https://farm4.staticflickr.com/3663/3392599156_e94f7d1098_z.jpg"

# load the model
model = OFAModel.from_pretrained(ckpt_dir, use_cache=False).to(device)
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

# load the image
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))

# Generate semantic masks
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize(img.size, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# extract CNN features from the model
with torch.no_grad():
    visual_embeds = model.encoder.embed_images(patch_resize_transform(img).unsqueeze(0).to(device))

visual_embeds = visual_embeds.detach().cpu().squeeze(0).permute(1, 2, 0)

# generate DFF semantic masks
out = generate_dff_masks(visual_embeds, k=10, img_size=img.size, mask_th=25, return_heatmaps=True)

# to visualize the masks run
# plot_masks(out['masks'])
```

The explainer expects a model that generates a caption given an image: ```model(img) --> caption```.
Therefore we write a simple wrapper for our model taking care of the preprocessing and the decoding required by the model.

```python3
class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, question, resolution, device="cpu"):
        super().__init__()
        
        self.resolution=resolution
        self.num_beams = 5
        self.no_repeat_ngram_size = 3
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.patch_resize_transform = transforms.Compose([
                                                            lambda image: image.convert("RGB"),
                                                            transforms.Resize(self.resolution, interpolation=Image.BICUBIC),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                        ])
        
        self.inputs = tokenizer([question], return_tensors="pt").input_ids.to(self.device)
        
    

    def forward(self, img):
        # put here all to code to generate a caption from an image
        
        patch_img = self.patch_resize_transform(img).unsqueeze(0).to(self.device)
        out_ids = model.generate(self.inputs, patch_images=patch_img, num_beams=self.num_beams, 
                                 no_repeat_ngram_size=self.no_repeat_ngram_size)
        
        return tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
```

Now we can generate a caption in this way 

```python3
question = "What is the subject doing?"
model_wrapper = ModelWrapper(model, tokenizer, question, resolution=img.size, device=device)
model_wrapper(img)
```

We have everything we need to create and run the KernelSHAP explainer.
```python3
explainer = BaseExplainer(model_wrapper, device=device)
shap, base = explainer.explain(img, out['masks'], k=-1)
```
We visualize the Shapley values corresponding to the visual features masks as a barchart, by running

```python3
labels = [ f"f_{i}" for i in range(shap.shape[0]) ]
barh(labels, shap)
```

and the visual explanation

```python3
heatmap(img, out['masks'], shap, alpha=0.75)
```
This will allow you to generate these sentence-based visual semantic explanations 
<img align="center" width="950" height="350" 
src="https://drive.google.com/uc?export=view&id=1HyxJ18wLKLEzMYg5fDpkBy3u88nS8-AG">



### Citation Information

```BibTeX
@article{cafagna2023interpreting,
  title={Interpreting Vision and Language Generative Models with Semantic Visual Priors},
  author={Cafagna, Michele and Rojas-Barahona, Lina M and van Deemter, Kees and Gatt, Albert},
  journal={arXiv preprint arXiv:2304.14986},
  year={2023}
}
```
