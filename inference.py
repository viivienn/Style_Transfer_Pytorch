from commons import get_model, load_image, im_convert
import torch
import torch.optim as optim
from PIL import Image
import numpy as np

vgg = get_model()

def get_prediction(content, style):
    try:
        content = load_image(content)
        style = load_image(style)

        content_features = get_features(content, vgg)
        style_features = get_features(style, vgg)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        style_weights = {'conv1_1': 1,
                         'conv2_1': 0.75,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}
        # alpha = content image weight
        # beta = style image wieght, lower the ratio, the more style is implemented in final image

        content_weight = 1
        style_weight = 1e6

        target = content.clone().requires_grad_(True)

        show_every = 1
        optimizer = optim.Adam([target], lr=0.003)
        steps = 1

        height, width, channels = im_convert(target).shape  # image as an rgb image
        image_array = np.empty(shape=(300, height, width, channels))

        capture_frame = steps / 300
        counter = 0

        for ii in range(1, steps + 1):
            target_features = get_features(target, vgg)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
            style_loss = 0

            for layer in style_weights:
                target_feature = target_features[layer]
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                _, d, h, w = target_feature.shape
                style_loss += layer_style_loss / (d * h * w)
            total_loss = content_loss * content_weight + style_weight * style_loss

            optimizer.zero_grad()  # reset optimizer
            total_loss.backward()  # how optimizer learns
            optimizer.step()  # update weight by iteration

            # if ii % show_every == 0:  # jumped 300 interation
            #     plt.imshow(im_convert(target))
            #     plt.axis('off')
            #     plt.show()
            # if ii % capture_frame == 0:
            #     image_array[counter] = im_convert(target)
            #     counter = counter + 1
        arr = im_convert(target) * 255
        arr = np.array(arr, dtype=np.uint8)
        return Image.fromarray(arr, 'RGB')
    except Exception as e:
        print(e)
        return 0, 'error'



def get_features(image, model):
    layers = {'0': 'conv1_1',  # more effective at recreating style fatures
              '4': 'conv2_1',
              '6': 'conv3_1',
              '10': 'conv4_1',
              '11': 'conv4_2',
              # content extraction, only one layer sufficient for content and deep into NN fro high depth image feature
              '12': 'conv5_1'}
    features = {}

    for name, layer in model._modules.items():
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features

def gram_matrix(tensor): #our tensor is 4d
  _, d, h, w = tensor.size() # batch, depth, height, weight
  tensor = tensor.view(d, h*w) #reshape into 2 dimensional tensor, maintain depth as number of feature channels
  gram = torch.mm(tensor, tensor.t()) #mm = multiply 2 tensors together, tensor.t = transpose
  return gram
