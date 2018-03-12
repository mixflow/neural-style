import torch
from torch.autograd import Variable
from torch import nn

import torchvision
from torchvision import transforms

from datetime import datetime
from PIL import Image



# the mean and std used for normalize image tensor. RGB three channels
# the number of mean and std come from https://github.com/pytorch/vision#models
MEAN = [0.5, 0.5, 0.5]# [ 0.485, 0.456, 0.406 ]
STD = [0.225, 0.225, 0.225]# [ 0.229, 0.224, 0.225 ]
dtype = torch.FloatTensor # cpu version. gpu: torch.cuda.FloatTensor

def load_image(img, size=512):
    # image to tensor with scale and normalization.
    transform = transforms.Compose([
        transforms.Resize(size), # The use of the transforms.Scale transform is deprecated (available torch 0.1.1)
        # image in range [0, 255] to a FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return transform(img).unsqueeze(0) # add addition dim, make batch data

def restore_image(var, will_be_image=True):
    # get image from tensor(Variable)
    # img = tensor[0]
    #
    # transform = transforms.Compose([
    #     # reverse the normalization( (x - mean) / std )
    #     transforms.Normalize(mean=[0, 0, 0], std=[1 / s for s in STD]),
    #     transforms.Normalize(mean=[-m for m in MEAN], std=[1, 1, 1]),
    #     transforms.ToPILImage()
    # ])
    # return transform(img)
    result = var[0].data* 0.225 + 0.5 # recover from normalization. (norm * std + mean = original)
    if will_be_image:
        pil_image_trans = transforms.ToPILImage()
        result = pil_image_trans(result)
    return result

# load Squeezenet pretrained model.
cnn = torchvision.models.squeezenet1_1(pretrained=True).features # without classifier layers
# cnn = torchvision.models.vgg19(pretrained=True).features
cnn.type(dtype)

# parameters are already trained.
for param in cnn.parameters():
    # neural style use the pretrained model and no need for updating the parameters
    param.requires_grad = False

def extract_features(x, cnn=cnn, layers=None):
    # squeezenet v1.1. 13 layers (indices 0 to 12)
    # 0 -> 5: conv - relu - maxpool - fire - fire - maxpool
    # 6 -> 12: fire - fire - maxpool - fire - fire - fire - fire

    features = []
    prev_feature = x

    if layers is not None: # only store the features on layers
        layer_iter = iter(layers)
        current_layer_idx = next(layer_iter)

    for idx, module in enumerate(cnn):
        next_feature = module(prev_feature)

        if layers is None: # no specified layers, store all features
            features.append(next_feature)
        elif current_layer_idx == idx:
            features.append(next_feature)
            # next layer idx
            current_layer_idx = next(layer_iter, None)
            if current_layer_idx is None: # no more layer
                break # stop for loop

        prev_feature = next_feature

    return features # concat features to one piece tensor

### Neural Style Loss
l2_loss = nn.MSELoss(size_average=False) # a function take two parameters: input, target. target requires_grad=False or volatile=True

def content_loss(current_image_features, content_features):
    '''
    Calculate the content loss (without weight multiplier)

    Parameters:
        current_image_features: features of current image (N, C, H, W). if single image, N=1
        content_features: features of content image

    Result:
        content_loss
    '''
    assert len(current_image_features) == len(content_features)

    losses = []
    # if extract multiply content layers, each layer has one loss, `content_losses` store those losses
    for idx, image in enumerate(current_image_features):
        content = content_features[idx]
        losses.append( l2_loss(image, content) )
        # losses.append( torch.sum( (content - image) ** 2 )  / 2 )

    return sum(losses) / len(content_features) # total lose

def gram_matrix(features, normalize=True):
    '''
    Calculate GramMatrix for style loss
    '''
    # if instanceof(features, list):
    #     # list to one piece tensor or variable
    #     features = torch.cat(list)

    n, c, h, w = features.size()
    # features_flat = features.view(n, c, h * w) # merge height and width dims
    # gram_matrix = torch.bmm(features_flat, features_flat.transpose(1, 2))
    features_flat = features.view(n * c, h * w) # merge height and width dims
    gram_matrix = features_flat @ features_flat.t() # @ is matrix multiply. same as the function torch.mm(x, y)

    if normalize:
        gram_matrix /= n * c * h * w

    return gram_matrix

def style_loss(current_image_features, style_gram_matrix, style_weights):

    losses = []
    num_features = len(style_gram_matrix)
    for idx, image in enumerate(current_image_features):
        image_gram = gram_matrix(image)
        style_gram = style_gram_matrix[idx]

        losses.append(l2_loss( image_gram, style_gram))
    # end for each image feature.

    total_loss = sum(losses).data[0]

    # total_loss / loss.data[0] * loss is normalize handle to prevent some layer loss too large or too small
    losses = [ style_weights[idx] * (total_loss / loss.data[0]) * loss for idx, loss in enumerate(losses)]
    result = sum(losses)

    # print("total loss vs result: ", total_loss / result.data[0], " times.", total_loss, result.data[0])
    return result

def add_path_before_ext(path, add_on):
    ext_idx = path.rindex('.')
    new_path = path[:ext_idx] + '-' + str(add_on) + path[ext_idx:]
    return new_path


def style_transfer(content_image, style_image, image_size,  output,
    content_layers,
    style_layers, style_weights,
    content_weight_factor=1e-1, style_weight_factor=1,
    style_size=224,
    init_random=True, num_iterations=500, save_image_interval=100, report_interval=20,
    ):
    '''
    merge content and style image.

    Parameters:
        content_image: filename(path) of content image (string)
        style_image: filename(path) of style image (string)
        image_size: size of smalleset image dim. used for content loss and output image.
        style_size_scale: style image size base on image size(scale * image_size). default 1

        content_layers: the content features layers which is extracted.
        content_weight: the weight on content loss
        style_layers: the style feature layers
        style_weights: the weight on each style layer loss

        init_random: if set True, use the noise image as initalize train image. else use content image.
        num_iterations: optimize steps.
        save_image_interval: the number of save image interval .
        report_interval: print current loss every interval.
        output: output image filename(path)


    '''

    # extract features of content image
    content_tensor = load_image( Image.open(content_image), size=image_size)
    content_variable = Variable(content_tensor.type(dtype), requires_grad=False)
    content_features = extract_features(content_variable, cnn=cnn, layers=content_layers)

    # extract features of style image
    style_tensor = load_image( Image.open(style_image), size=style_size)
    style_variable =  Variable(style_tensor.type(dtype), requires_grad=False)
    style_features = extract_features(style_variable, cnn=cnn, layers=style_layers)
    style_gram_matrix = [ gram_matrix(f) for f in style_features]

    #  init output image(the train image)
    if init_random: # noise init image
        image_tensor = torch.Tensor(content_variable.size()).type(dtype)
    else: # copy content image
        image_tensor = content_tensor.clone().type(dtype)

    # optimize output image, need gradient computation.
    image_variable = Variable(image_tensor, requires_grad=True)

    # hyperparmaters
    learning_rate = 0.35# 3.0
    decayed_lr = 0.1
    decay_lr_at = 300

    decayed_lr_2 = 0.02
    decay_lr_at_2 = 600

    optimizer = torch.optim.Adam([image_variable], lr=learning_rate)
    start_time = datetime.now() # record time to calculate time cost

    loss_history = []
    best_loss = float('inf')
    best_image_variable = None

    for t in range(num_iterations):


        optimizer.zero_grad()
        ''' !! clamp scope must be based on data scope which is normalizated (mean, std)

            if mean is 0.5, std is 0.5. scream_content13the original data is in [0,1].
            after nomalization, the data is in [-1, 1] ( data - mean / std)
        '''
        # if t <= 180 or t >= num_iterations*0.75:
            # mean 0.5, std 0.225, data range [0, 1]
            # so [0 - 0.5, 1 - 0.5] / 0.225 ~= [-2.22, 2.22]
            # the multiplier is the extend factor. if only need the exact clamp, this factor would be 1.
            # but I need addition space to prevent cutting image data range too much.
            # image_tensor.clamp_(-2.22 * 1, 2.22 * 1)
        image_tensor.clamp_(-2.22 * 1, 2.22 * 1)

        image_feature_last_layer = max( max(content_layers), max(style_layers)) # the last layer number
        image_features = extract_features(image_variable, cnn, list(range(image_feature_last_layer + 1)))


        # calculate content and style loss
        c_loss = content_loss([image_features[x] for x in content_layers], content_features)
        c_loss = content_weight_factor * c_loss
        s_loss = style_loss([image_features[y] for y in style_layers], style_gram_matrix, style_weights)
        s_loss = style_weight_factor * s_loss

        loss = c_loss + s_loss

        current_loss = loss.data[0] # float number
        loss_history.append(current_loss)

        if current_loss < best_loss:
            best_image_variable = image_variable.clone()

        loss.backward()

        if t == decay_lr_at:
            optimizer = torch.optim.Adam([image_variable], lr=decayed_lr)
        elif t == decay_lr_at_2:
            optimizer = torch.optim.Adam([image_variable], lr=decayed_lr_2)
        optimizer.step()


        if t % report_interval == 0:
            print('Iteration: ', t, ' loss: ', "{:.6g}".format(current_loss),
                'c_loss', "{:.6g}".format(c_loss.data[0]),
                's_loss', "{:.6g}".format(s_loss.data[0]),
                '  time per step: ',
                (datetime.now() - start_time) / report_interval)
            start_time = datetime.now()

        if t % save_image_interval == 0 and t != 0:
            mid_output_name = add_path_before_ext(output, t)

            # torchvision.utils.save_image(image_tensor, mid_output_name)
            print("saved image path: ", mid_output_name)
            restore_image(image_variable).save(mid_output_name)

    # save output image
    # torchvision.utils.save_image(image_tensor, output)225
    print("saved final image path: ", output)
    restore_image(image_variable).save(output)

    restore_image(best_image_variable).save(add_path_before_ext(output, '-best'))

if __name__ == '__main__':

    # the parameters , settings
    ALSACE_starry = {
        'content_image' : 'images/ALSACE.jpg',
        'style_image' : 'images/starry_night.jpg',
        'output' : 'output/ALSACE_starry_slayer2x2.jpg',
        'image_size' : 512,
        'style_size' : 224,
        'content_layers' : [1,2,3],
        'content_weight_factor' :1e-1,
        'style_layers' : [1, 2, 3, 4, 6, 7, 9],
        'style_weights' : [21,21,1,1,1,7,7],
        'num_iterations' : 600,
    }

    # actually call the nerual style transfer process.
    style_transfer(**ALSACE_starry)
