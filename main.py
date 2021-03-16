# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import argparse
import cv2
import os
import time
import numpy

from dataset import LabelMeDataset
from tool import mask2img, draw_results

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default="/home/dominique/Schreibtisch/Seg_Code/label/Segmented_Images/FIN18json")
    parser.add_argument('--out', type=str,
                        default="/home/dominique/Schreibtisch/Seg_Code/label/Segmented_Images/FIN18png")
    parser.add_argument('--augmentation', action='store_true', default=False)
    return parser.parse_args()


to_image = transforms.ToPILImage()

if __name__ == '__main__':
    args = parse_args()
    dataset = LabelMeDataset(args.dataset, args.augmentation)

    num_samples = len(dataset)
    inf = "/home/dominique/Schreibtisch/Seg_Code/label/Segmented_Images/FIN18"
    for idx in range(num_samples):
        image, mask = dataset[idx]

        # torch.set_printoptions(profile="full")   # (threshold=10_000)  # (edgeitems=3)  # (profile="full")
        # print(mask[:, 300:400, 300:400])
        mask = mask > 0

        label_png = mask2img(mask, image.shape, categories=dataset.categories).cpu().numpy()

        # Generate Dataset in Cityscape style
        # cv2.imwrite(str(dataset.newpath(idx, args.out, ".png")), label_png)
        # if idx % validation_size != 0:
        #     validation.write(str(dataset.newpath(idx, inf, ".png")) + "," + str(dataset.newpath(idx, args.out, ".png")) + "\r\n")
        # else:
        #     train.write(str(dataset.newpath(idx, inf, ".png")) + "," + str(dataset.newpath(idx, args.out, ".png")) + "\r\n")

        # image = to_image(image).convert("RGB") transforms.ToPILImage()
        # image.show()
        #
        # time.sleep(5)
        # os.system('pkill eog') #if you use GNOME Viewer
        # image.close()

        image = image.cpu().numpy()
        image = numpy.transpose(image, (1, 2, 0))
        img_mean=(0.485, 0.456, 0.406)
        img_std=(0.229, 0.224, 0.225)
        image = (image * img_std) + img_mean
        image = (255 * image).astype(numpy.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("heh", label_png)
        cv2.imshow("hier", image)

        # for category, category_image, mask_image in draw_results(image, mask, categories=dataset.categories):
        #     cv2.imshow(category+" masked", category_image)
        #     cv2.imshow(category, mask_image)

        if cv2.waitKey(0) == ord('q'):
            exit()
