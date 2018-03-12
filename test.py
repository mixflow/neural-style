import unittest
from unittest.mock import Mock

from neuralstyle import load_image, restore_image, extract_features, content_loss, gram_matrix, style_loss
from PIL import Image

import torch, torchvision
from torchvision import transforms
from torch.autograd import Variable

import numpy as np

class TestImageMethods(unittest.TestCase):

    def test_load(self):
        with Image.open("images/starry_night.jpg") as img:
            source_width, source_height = img.size

            result = load_image(img,size=512)
            _, C, H, W = result.size()
            self.assertEqual(C, 3)
            self.assertTrue(H/W - source_width/source_height <= 0.001)

    def test_restore_image(self):
        scale_tensor_trans = transforms.Compose([
            transforms.Scale(512),
            transforms.ToTensor()
        ])
        with Image.open("images/starry_night.jpg") as img:
            original_image_tensor = scale_tensor_trans(img)

            image = load_image(img, size=512)
            # no need transform PIL image. we need the tensor data
            result = restore_image(Variable(image, requires_grad=False), will_be_image=False)
            _, height, width = result.size()
            self.assertEqual(height, 512)

            # is the restore image data equal the original one.
            all_one = torch.ones(3, height, width).type(torch.ByteTensor)
            error = (original_image_tensor - result <= 1e-5)
            self.assertTrue(torch.equal(all_one,error))

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def features_from_img(imgpath, imgsize, layers=None):
    img = load_image(Image.open(imgpath), size=imgsize)
    img_var = Variable(img.type(dtype))
    return extract_features(img_var, cnn, layers), img_var

answers = np.load("style-transfer-checks.npz") # correct loss data

dtype = torch.FloatTensor
cnn = torchvision.models.squeezenet1_1(pretrained=True).features # without classifier layers
cnn.type(dtype)

class TestNeuralStyleMethods():

    def test_extract_features(self):
         mock = Mock()

         # mock._modules.values.return_value = iter([lambda x:x+1] * 10)
         mock.return_value = iter([lambda x:x+1] * 10)
         features = extract_features(1, mock, layers=[0,1,3,4,7])
         self.assertEqual(features, [2,3,5,6,9])

    def test_content_loss(self):
        content_image = 'images/tubingen.jpg'
        image_size =  192
        content_layers = [3]
        content_weights = [6e-2]

        c_feats, content_img_var = features_from_img(content_image, image_size, content_layers)

        bad_img = Variable(torch.zeros(*content_img_var.data.size()))
        image_feats = extract_features(bad_img, cnn, content_layers)

        output = content_loss(image_feats, c_feats, content_weights).data.numpy()
        error = rel_error(answers['cl_out'], output)
        self.assertTrue(error <= 0.001)

    def test_gram_matrix(self):
        style_image = 'images/starry_night.jpg'
        style_size = 192
        feats, _ = features_from_img(style_image, style_size)

        output = gram_matrix(feats[5].clone()).data.numpy()
        error = rel_error(answers['gm_out'], output)
        self.assertTrue(error <= 0.001)

    def test_style_loss(self):
        content_image = 'images/tubingen.jpg'
        style_image = 'images/starry_night.jpg'
        image_size =  192
        style_size = 192
        style_layers = [1, 4, 6, 7, 10]
        style_weights = [300000, 1000, 15, 3, 0]

        c_feats, _ = features_from_img(content_image, image_size, style_layers)
        feats, _ = features_from_img(style_image, style_size, style_layers)

        output = style_loss(c_feats, feats, style_weights).data.numpy()
        error = rel_error(answers['sl_out'], output)
        self.assertTrue(error <= 0.001)


if __name__ == '__main__':
    unittest.main()
