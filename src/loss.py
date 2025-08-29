import torch
from torchvision import transforms, models
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable

# Define style weights for different layers
STYLE_WEIGHTS = {
    'relu1_2': 1.0 / 2.6,
    'relu2_2': 1.0 / 4.8,
    'relu3_3': 1.0 / 3.7,
    'relu4_3': 1.0 / 5.6,
    'relu5_3': 10.0 / 1.5
}

def get_features(image, model, layers=None):
    """
    Extract features from specific layers of a model for a given image.

    Args:
        image (torch.Tensor): Input image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).
        layers (dict): Mapping of layer indices to layer names.

    Returns:
        dict: A dictionary of features for the specified layers.
    """
    if layers is None:
        layers = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
            '29': 'relu5_3'
        }

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    """
    Compute the Gram matrix for a given tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, depth, height, width).

    Returns:
        torch.Tensor: Gram matrix of the input tensor.
    """
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)  # Reshape tensor for matrix multiplication
    gram = torch.mm(tensor, tensor.t())  # Compute Gram matrix
    return gram

def gram_loss(style, target, model):
    """
    Compute the Gram loss (style loss) between a style image and a target image.

    Args:
        style (torch.Tensor): Style image tensor.
        target (torch.Tensor): Target image tensor.
        model (torch.nn.Module): Pretrained model (e.g., VGG).

    Returns:
        torch.Tensor: The computed Gram loss.
    """
    # Extract features for the style and target images
    style_features = get_features(style, model)
    target_features = get_features(target, model)

    # Compute Gram matrices for the style image
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Initialize total loss
    total_loss = 0

    # Compute the weighted Gram loss for each layer
    for layer, weight in STYLE_WEIGHTS.items():
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]

        # Compute the layer-specific Gram loss
        _, d, h, w = target_feature.shape
        layer_loss = weight * torch.mean((target_gram - style_gram) ** 2)
        total_loss += layer_loss / (d * h * w)

    return total_loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2, mask=None):
    """
    img1, img2: ((B), C, H, W)
    mask: (B, 1, H, W)
    """
    if len(img1.shape) == 3:
        # (C, H, W) -> (H, W, C)
        img1 = img1.permute(1, 2, 0)
        img2 = img2.permute(1, 2, 0)
    elif len(img1.shape) == 4:
        # (B, C, H, W) -> (B, H, W, C)
        img1 = img1.permute(0, 2, 3, 1)
        img2 = img2.permute(0, 2, 3, 1)

    if mask is not None:
        mask = mask.squeeze(0)
        img1 = img1[mask]
        img2 = img2[mask]
    # mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(dim=0, keepdim=True)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True, stride=1):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, stride)


def _ssim(img1, img2, window, window_size, channel, size_average=True, stride=1):
    mu1 = F.conv2d(
        img1, window, padding=window_size // 2, groups=channel, stride=stride
    )
    mu2 = F.conv2d(
        img2, window, padding=window_size // 2, groups=channel, stride=stride
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=window_size // 2,
            groups=channel,
            stride=stride,
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=window_size // 2,
            groups=channel,
            stride=stride,
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=window_size // 2,
            groups=channel,
            stride=stride,
        )
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
