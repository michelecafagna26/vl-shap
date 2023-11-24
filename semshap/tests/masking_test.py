import unittest

from semshap.masking import genenerate_vit_masks, generate_dff_masks, generate_segmentation_masks
import requests
from io import BytesIO
from PIL import Image
import torch


class MaskingTest(unittest.TestCase):
    # def test_generate_dff_masks(self):
    #     self.assertEqual(True, False)  # add assertion here

    def test_genenerate_vit_masks(self):

        last_hidden_state = torch.rand((197, 768))
        img_size = (128, 128)

        out = genenerate_vit_masks(last_hidden_state, img_size, k=10, return_heatmaps=True)

        self.assertEqual(img_size, out['masks'][0].shape)
        self.assertEqual(img_size, out['masks'][-1].shape)
        self.assertEqual(11, len(out['masks']))

    def test_genenerate_dff_masks(self):

        fake_embeds = torch.rand((32, 32, 1024))
        img_size = (128, 128)

        out = generate_dff_masks(fake_embeds, k=10, img_size=img_size, mask_th=25, return_heatmaps=True)

        self.assertEqual(img_size, out['masks'][0].shape)
        self.assertEqual(img_size, out['masks'][-1].shape)
        self.assertEqual(10, len(out['masks']))

    def test_genenerate_segmentation_masks(self):

        img = Image.open("./semshap/tests/assets/moto1.jpg")
        prompts = ['person', 'motorcycle', 'dirt', 'sky']
        out = generate_segmentation_masks(img, prompts, img_size=img.size, )

        self.assertEqual(img.size, out['masks'][0].shape)
        self.assertEqual(img.size, out['masks'][-1].shape)
        self.assertEqual(5, len(out['masks']))


if __name__ == '__main__':
    unittest.main()
