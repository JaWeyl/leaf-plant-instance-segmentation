""" Configuration file.
"""
import copy

args = dict(
  train_img_dir='./dataset-mini/images/train',

  val_img_dir='./dataset-mini/images/val',
  val_anno_dir='./dataset-mini/annos/val',

  test_img_dir='./dataset-mini/images/test',

  report_dir = './logs/reports',

  width = 1024,
  height = 512,

  n_classes = 1,
  n_sigma = 3,
  apply_offsets = True,

  sigma_scale = 11.0,
  alpha_scale = 11.0,

  parts_area_thres=32,
  parts_score_thres=0.9,

  objects_area_thres = 64,
  objects_score_thres = 0.9,

  cls_colors = {"0" : "#ff0000", "1" : "#1eff00"}
)

def get_args():
  return copy.deepcopy(args)