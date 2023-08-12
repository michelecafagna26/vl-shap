import unittest

from semshap.masking import genenerate_vit_masks
import requests
from io import BytesIO
from PIL import Image
import torch


class MaskingTest(unittest.TestCase):
    # def test_generate_dff_masks(self):
    #     self.assertEqual(True, False)  # add assertion here

    def test_genenerate_vit_masks(self):

        img_url = "https://marhamilresearch4.blob.core.windows.net/stego-public/sample_images/moto1.jpg"
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))

        last_hidden_state = torch.rand((197, 768))

        out = genenerate_vit_masks(last_hidden_state, img.size, k=10, return_heatmaps=False)

        self.assertEqual(len(out['masks']), 11)
        self.assertEqual(out['masks'][0].shape, img.size)

if __name__ == '__main__':
    unittest.main()
