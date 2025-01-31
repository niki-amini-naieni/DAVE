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
exemplars = {
    
}
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
    abs_errs = []

    for img_f_name in img_names:
        image = Image.open(img_f_name)
        width, height = image.size
        # Get exemplars.
        if "ma
        
        bounding_boxes = []
        for i in range(2):
          if i == 0:
              idx = random.randint(0, int(data['boxes'].shape[1] / 2))
          else:
              idx = random.randint(int(data['boxes'].shape[1] / 2) - 1, data['boxes'].shape[1] - 1)

          box = data['boxes'][0][i]
          box2 = [int(k) for k in box]
          x1, y1, x2, y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
          bounding_boxes.append((x1, y1, x2, y2))

        bboxes = torch.tensor(bounding_boxes)

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
            score = predicted_bboxes.fields['scores'][i]
            pred_annotations['annotations'].append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x0, y0, box_w, box_h],
                    "score": score.item(),
                    "id": pred_anno_id,
                    "area": (box_w * box_h)
                }
            )
            pred_anno_id +=1
        img_id +=1
            
        pred_count = round(denisty_map.sum().item(), 1)

        print("Pred. Count: " + str(pred_count) + ", GT Count: " + str(gt_count))
        err = abs(pred_count - gt_count)
        abs_errs.append(err)

    abs_errs = np.array(abs_errs)
    print("Number of Images Tested: " + str(len(abs_errs)))
    print("MAE: " + str(np.mean(abs_errs)))
    print("RMSE: " + str(np.sqrt(np.mean(abs_errs**2))))

    with open("dave_box_preds_carpk.json", 'w') as out_file:
        json.dump(pred_annotations, out_file)

    with open("carpk_coco.json", 'w') as out_file:
        json.dump(gt_annotations, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
