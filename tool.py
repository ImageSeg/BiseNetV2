from typing import List
import torch
import cv2
import numpy
from PIL import Image


def draw_results(
        image: torch.Tensor,
        mask: torch.Tensor,
        categories: List[str],
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225)
):
    assert mask.shape[0] == len(categories)
    assert image.shape[1:] == mask.shape[1:]
    assert mask.dtype == torch.bool

    image = image.cpu().numpy()
    image = numpy.transpose(image, (1, 2, 0))
    image = (image * img_std) + img_mean
    image = (255 * image).astype(numpy.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = mask.cpu().numpy()

    colours = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    ]

    for label, (category, category_mask) in enumerate(zip(categories, mask)):
        cat_image = image.copy()
        cat_image[category_mask] = 0.5 * cat_image[category_mask] + 0.5 * numpy.array(
            colours[categories.index(category)])

        mask_image = image.copy()
        mask_image[~category_mask] = 0

        yield category, cat_image, mask_image


def mask2img(
        mask: torch.Tensor,
        shape: torch.Tensor,
        categories: List[str]
):
    assert mask.shape[0] == len(categories)
    assert shape[1:] == mask.shape[1:]
    assert mask.dtype == torch.bool
    c, h, w = shape

    colours = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    ]

    label_png = torch.zeros((h, w, c))
    for (category, category_mask) in zip(categories, mask):
        label_png[category_mask] += torch.tensor(colours[categories.index(category)])

    return label_png


# helper function to show an image
# (used in the `plot_classes_preds` function below)
# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))

cfg = dict(
    num_aux_heads=4,
    lr_start=5e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=150000,
    im_root='./datasets/cityscapes',
    train_im_anns='./datasets/cityscapes/train.txt',
    val_im_anns='./datasets/cityscapes/val.txt',
    scales=[0.25, 2.],
    cropsize=[512, 1024],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)


class Dict:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def concatenate_image(image1, image2):
    images = [image1, image2]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def class2image(
        class_tensor: torch.Tensor,
        categories: List[str]
):
    assert len(class_tensor.unique()) == (1+len(categories))
    h, w = class_tensor.shape

    colours = [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    ]

    class_image = torch.zeros((h, w, 3), dtype=torch.uint8)
    for category in categories:
        class_image[class_tensor == (1+categories.index(category))] = torch.tensor(colours[categories.index(category)], dtype=torch.uint8)

    return class_image
