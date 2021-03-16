# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
import cv2
import os
import time
import torch
from PIL import Image
from archt import BiSeNetV2

from dataset import LabelMeDataset
from tool import mask2img, draw_results, concatenate_image, class2image

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="/home/dominique/Bilder/source/FIN18_ss19lat_pov_rate25")
    parser.add_argument('--out', type=str, default="")
    parser.add_argument('--path', type=str, default="model2.pth")
    parser.add_argument('--augmentation', action='store_true', default=False)
    return parser.parse_args()


to_image = transforms.ToPILImage()

# palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette = torch.tensor([
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255),
        (0, 255, 128), (128, 0, 255)
    ], dtype=torch.uint8)


if __name__ == '__main__':
    args = parse_args()
    dataset = LabelMeDataset(args.dataset, args.augmentation)

    model = BiSeNetV2(n_classes=dataset.un_class)    # , output_aux=False)
    model.load_state_dict(torch.load(args.path, map_location='cpu'))
    model.eval()

    num_samples = len(dataset)

    # for idx in range(num_samples):
    #    image, mask = dataset[idx]
    for idx, (image, mask) in enumerate(dataset):
        prediction = model(image.unsqueeze(0))[0]
        prediction = prediction.argmax(dim=1).squeeze().detach().cpu().numpy()

        prediction = palette[prediction]
        mask = palette[mask]

        # show image|label
        image = to_image(image).convert("RGB")
        mask = to_image(mask.cpu().numpy()).convert("RGB")
        prediction = to_image(prediction.cpu().numpy()).convert("RGB")

        show_image = Image.blend(image, prediction, alpha=0.5)  # concatenate_image(prediction, image)   #
        show_image.show()
        time.sleep(5)
        os.system('pkill eog')  # if you use GNOME Viewer
        show_image.close()
