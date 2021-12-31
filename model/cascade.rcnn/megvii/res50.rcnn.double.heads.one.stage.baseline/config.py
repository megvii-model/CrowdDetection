import os, getpass
import sys
import os.path as osp
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../../..'
add_path(os.path.join(root_dir))
add_path(os.path.join(root_dir, 'lib'))
add_path(osp.join(root_dir, 'util'))
class Crowdhuman:
    class_names = ['background', 'person']
    num_classes = len(class_names)
    root_folder = '/home/zhenganlin/june/CrowdHuman'
    image_folder = osp.join(root_folder, 'images')
    train_source = osp.join(root_folder, 'crowd_human_train15000_final_unsure_fixempty_fixvis_vboxmerge.odgt')
    eval_source = osp.join(root_folder, 'crowd_human_test4370_final_unsure_fixempty_fixvis_vboxmerge.odgt')

class Config:

    usr = getpass.getuser()
    this_model_dir = osp.split(os.path.realpath(__file__))[0].split('/')[-1]
    output_dir = osp.join(root_dir, 'output', usr, 'fpn', 'human', this_model_dir)
    model_dir = osp.join(output_dir, 'model_dump')
    eval_dir = osp.join(output_dir, 'eval_dump')
    log_dir = output_dir
    logger = osp.join(output_dir, 'logger.log')
    init_weights = '/home/zhenganlin/june/CrowdHuman/resnet50_fbaug.pth'
    # ----------data config---------- #
    image_mean = np.array([103.530, 116.280, 123.675])
    image_std = np.array([57.375, 57.120, 58.395])
    train_image_short_size = 800
    train_image_max_size = 1400
    eval_resize = True
    eval_image_short_size = 800
    eval_image_max_size = 1400
    seed_dataprovider = 3
    datadb = Crowdhuman
    train_source = datadb.train_source
    eval_source = datadb.eval_source
    eval_json, train_json = eval_source, train_source
    image_folder = datadb.image_folder
    imgDir = image_folder
    class_names = datadb.class_names
    num_classes = datadb.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    backbone_freeze_at = 2
    rpn_channel = 256
    
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 1e-3 * 1.25

    warm_iter = 800
    max_epoch = 35
    lr_decay = [24, 27]
    nr_images_epoch = 15000
    log_dump_interval = 1
    iter_per_epoch = nr_images_epoch // train_batch_per_gpu

    # ----------test config---------- #
    test_nms = 0.5
    test_nms_method = 'set_nms'
    visulize_threshold = 0.3
    pred_cls_threshold = 0.05
    nr_info_dim=6
    # ----------model config---------- #
    batch_filter_box_size = 0
    nr_box_dim = 5
    ignore_label = -1
    max_boxes_of_image = 500
    num_predictions = 2
    # ----------rois generator config---------- #
    anchor_base_size = 32
    anchor_base_scale = [1]
    anchor_aspect_ratios = [1, 2, 3]
    num_cell_anchors = len(anchor_aspect_ratios)
    anchor_within_border = False
    anchor_ignore_label = -1

    rpn_min_box_size = 2
    rpn_nms_threshold = 0.7
    train_prev_nms_top_n = 12000
    train_post_nms_top_n = 2000
    test_prev_nms_top_n = 6000
    test_post_nms_top_n = 1500

    # ----------binding&training config---------- #
    rpn_smooth_l1_beta = 1
    rcnn_smooth_l1_beta = 1

    num_sample_anchors = 256
    positive_anchor_ratio = 0.5
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_bbox_normalize_targets = False

    num_rois = 512
    fg_ratio = 0.5
    fg_threshold = 0.5
    bg_threshold_high = 0.5
    bg_threshold_low = 0.0
    rcnn_bbox_normalize_targets = True
    bbox_normalize_means = np.array([0, 0, 0, 0])
    bbox_normalize_stds = np.array([0.1, 0.1, 0.2, 0.2])

config = Config()

