""" Perform automated postprocessing step to cluster individual crop leaf and plant instances.
"""
import os

import cv2
import h5py
import numpy as np
import skimage.exposure
import skimage.io

import report_config
from utils.myutils import (Cluster, Visualizer, bounding_box_from_mask,
                           get_all_predictions)

args = report_config.get_args()

path_train_reports = os.path.join(args["report_dir"], 'train')
path_val_reports = os.path.join(args["report_dir"], 'val')
path_test_reports = os.path.join(args["report_dir"], 'test')

train_preds = get_all_predictions(path_train_reports)
val_preds = get_all_predictions(path_val_reports)
test_preds = get_all_predictions(path_test_reports)

cluster = Cluster('np',
                  args['width'],
                  args['height'],
                  args['n_classes'],
                  args['n_sigma'],
                  args['sigma_scale'],
                  args['alpha_scale'],
                  args['parts_area_thres'],
                  args['parts_score_thres'],
                  args['objects_area_thres'],
                  args['objects_score_thres'],
                  args['apply_offsets'])

vis = Visualizer(args['train_img_dir'], args['val_img_dir'], args['test_img_dir'], args['n_classes'], args['cls_colors'], args['width'], args['height'])

# for train_pred in train_preds:
#   print("=> report train ", train_pred.path, train_pred.pred)
#   vis.set_status('train')

#   pred = train_pred.load()
#   objects_seed, parts_seed, objects_offsets, parts_offsets, objects_sigma, parts_sigma, results = cluster.cluster(pred)

#   vis.plot_sigmas(train_pred.path, train_pred.pred, objects_sigma, parts_sigma)
#   vis.plot_seed(train_pred.path, train_pred.pred, objects_seed, parts_seed)
#   vis.plot_embeddings(train_pred.path, train_pred.pred, results, parts_offsets, objects_offsets)
#   vis.plot_instances(train_pred.path, train_pred.pred, results)

epoch = ''
for val_pred in val_preds:
  print("=> report val ", val_pred.path, val_pred.pred)
  vis.set_status('val')
  
  img_name = val_pred.pred.split(".")[0] # filename of current image
  current_epoch = val_pred.path.split('/')[-1] # e.g. '0127'
  if current_epoch != '-001':
    continue
  if current_epoch != epoch:
    epoch = current_epoch

    if 'hdf5_ground_truth' in locals():
      hdf5_ground_truth.close()
    if 'hdf5_predictions' in locals():
      hdf5_predictions.close()    

    # create export directories for current epoch
    export_dir_gt = os.path.join(path_val_reports, epoch, 'patches', 'ground_truth')
    if not os.path.exists(export_dir_gt):
      os.makedirs(export_dir_gt)
    
    export_dir_pred = os.path.join(path_val_reports, epoch, 'patches', 'pred')
    if not os.path.exists(export_dir_pred):
      os.makedirs(export_dir_pred)

    # create hdf5 files to store ground truth and predictions patches
    hdf5_ground_truth = h5py.File(os.path.join(export_dir_gt, 'ground_truth.h5'), 'w')
    hdf5_predictions = h5py.File(os.path.join(export_dir_pred, 'predictions.h5'), 'w')

  pred = val_pred.load()
  objects_seed, parts_seed, objects_offsets, parts_offsets, objects_sigma, parts_sigma, results = cluster.cluster(pred)

  vis.plot_sigmas(val_pred.path, val_pred.pred, objects_sigma, parts_sigma)
  vis.plot_seed(val_pred.path, val_pred.pred, objects_seed, parts_seed)
  vis.plot_embeddings(val_pred.path, val_pred.pred, results, parts_offsets, objects_offsets)
  vis.plot_instances(val_pred.path, val_pred.pred, results)

  part_map = cluster.draw_part_map(results, cls_idx='0')
 
  f_global_anno = os.path.join(args['val_anno_dir'], 'global', img_name + ".semantic")
  global_anno = np.fromfile(f_global_anno, dtype=np.uint32)
  global_anno = global_anno.reshape(args['height'], args['width'])
  global_instance_ids = np.unique(global_anno)
  if 0 in global_instance_ids:
    global_instance_ids = global_instance_ids[1:]
 
  f_parts_anno = os.path.join(args['val_anno_dir'], 'parts', img_name + ".semantic")
  part_anno = np.fromfile(f_parts_anno, dtype=np.uint32)
  part_anno = part_anno.reshape(args['height'], args['width'])
 
  for idx_, id_ in enumerate(global_instance_ids):
    gt_mask = np.zeros((args['height'], args['width']), dtype=np.uint8) # canvas
    
    instance_mask = (global_anno == id_)
    x_tl, y_tl, w, h = bounding_box_from_mask(instance_mask)
 
    part_ids = np.unique(part_anno[instance_mask])
    if 0 in part_ids:
      part_ids = part_ids[1:]
 
    for count, part_id in enumerate(part_ids):
      part_mask = (part_anno == part_id).astype(np.uint8)
      gt_mask += (part_mask * (count + 1))
 
    gt_mask = gt_mask[y_tl: (y_tl + h), x_tl: (x_tl + w)]
    pred_mask = part_map[y_tl: (y_tl + h), x_tl: (x_tl + w)]
 
    img_patch_name = img_name + '_{0:04d}'.format(idx_) # filename of current patch
    f_gt_export = os.path.join(export_dir_gt, img_patch_name)
    
    hdf5_ground_truth_img_group = hdf5_ground_truth.create_group('Dataset/' + img_patch_name)
    hdf5_ground_truth_img_group.create_dataset("label", data=gt_mask)
    hdf5_ground_truth_img_group.create_dataset("label_filename", data=img_patch_name)
    skimage.io.imsave(f_gt_export + '.png', skimage.exposure.rescale_intensity(gt_mask), check_contrast=False)
 
    f_pred_export = os.path.join(export_dir_pred, img_patch_name)
    
    hdf5_predictions_img_group = hdf5_predictions.create_group('Dataset/' + img_patch_name)
    hdf5_predictions_img_group.create_dataset("label", data=pred_mask)
    hdf5_predictions_img_group.create_dataset("label_filename", data=img_patch_name) 
    skimage.io.imsave(f_pred_export + '.png', skimage.exposure.rescale_intensity(pred_mask), check_contrast=False)

if 'hdf5_ground_truth' in locals():
  hdf5_ground_truth.close()
if 'hdf5_predictions' in locals():
  hdf5_predictions.close()


  

