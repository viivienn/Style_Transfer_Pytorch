import numpy as np
from PIL import Image
from torchvision import transforms, models


def get_model():
    vgg = models.squeezenet1_0(pretrained=True).features

    for param in vgg.parameters():
        param.requires_grad_(False)

    return vgg

def im_convert(tensor):
  image = tensor.cpu().clone().detach().numpy()
  #remove single dimensional entries from the shape of the image from the shape of the array
  image = image.squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
  image = image.clip(0, 1)
  return image

def load_image(image, max_size=400, shape=None):
    image = Image.open(image).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)  # leave size of image unchanged if smaller than threshold value

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    image = in_transform(image).unsqueeze(0)
    return image

