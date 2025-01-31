from torch.nn import DataParallel
from models.dave import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
import os
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import random
from utils.data import resize
import matplotlib.pyplot as plt
import hub
import torchvision.transforms.functional as F
#from torchvision import Transforms as T

# fix the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

"""
def on_click(event):
    # Record the starting point of the bounding box
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # Connect the release event
    fig.canvas.mpl_connect('button_release_event', on_release)


def on_release(event):
    # Record the ending point of the bounding box
    global ix, iy
    x, y = event.xdata, event.ydata
    # Calculate the width and height of the bounding box
    width = x - ix
    height = y - iy
    # Add a rectangle patch to the axes
    rect = patches.Rectangle((ix, iy), width, height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Store the bounding box coordinates
    bounding_boxes.append((ix, iy, ix + width, iy + height))
    plt.draw()

"""
exemplar_image_names = {"ma2035_023": "ma2035_023_internal_exemplar_image.jpg", "ma2035_024": "ma2035_024_internal_exemplar_image.jpg", "ma2035_039": "ma2035_039_internal_exemplar_image.jpg", "ma2035_040": "ma2035_040_internal_exemplar_image.jpg", "ma2035_069": "ma2035_069_internal_exemplar_image.jpg", "ma2035_072": "ma2035_072_internal_exemplar_image.jpg", "ma2035_169": "ma2035_169_internal_exemplar_image.jpg"}
exemplar_bbox_file_names = {"ma2035_023": "ma2035_023_internal_exemplars.json", "ma2035_024": "ma2035_024_internal_exemplars.json", "ma2035_039": "ma2035_039_internal_exemplars.json", "ma2035_040": "ma2035_040_internal_exemplars.json", "ma2035_069": "ma2035_069_internal_exemplars.json", "ma2035_072": "ma2035_072_internal_exemplars.json", "ma2035_169": "ma2035_169_internal_exemplars.json"}
def list_files_recursive(path='.'):
    anno_json = {}
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        elif ".jpg" in entry:
            anno_json[full_path] = []
    return anno_json

@torch.no_grad()
def demo(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    img_names = list_files_recursive(args.video_dir)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_3_shot.pth'))['model'], strict=False
    )
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    model.eval()

    print(img_names)

    # Make predictions.
    pred_boxes = {}

    for img_f_name in img_names:
        image = Image.open(img_f_name)
        width, height = image.size
        # Get exemplars.
        for sequence in exemplar_image_names:
            if sequence in img_f_name:
                exemplar_image_name = exemplar_image_names[sequence]
                exemplar_bbox_file_name = exemplar_bbox_file_names[sequence]
                print("Using image: " + str(img_f_name))
                print("Using exemplar image: " + str(exemplar_image_name))
                print("Using exemplars: " + str(exemplar_bbox_file_name))
                break
        exemplar_image = Image.open(exemplar_image_name)
        with open(exemplar_bbox_file_name) as exemplar_file:
            exemplar_data = json.load(exemplar_file)
        exemplars = []
        for exemplar in exemplar_data:
            x0, y0, x1, y1 = exemplar[0], exemplar[1], exemplar[3], exemplar[4]
            exemplars.append((x0, y0, x1, y1))

        bboxes = torch.tensor(exemplars)

        img, bboxes, scale = resize(image, bboxes)
        print("img shape: " + str(img.shape))
        img = img.unsqueeze(0).to(device)
        bboxes = bboxes.unsqueeze(0).to(device)

        denisty_map, _, tblr, predicted_bboxes = model(img, bboxes=bboxes)
        pred_boxes = predicted_bboxes.box.cpu() / torch.tensor([scale[0], scale[1], scale[0], scale[1]])
        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            x0, y0, box_w, box_h = xmin.item(), ymin.item(), (xmax - xmin).item(), (ymax - ymin).item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
