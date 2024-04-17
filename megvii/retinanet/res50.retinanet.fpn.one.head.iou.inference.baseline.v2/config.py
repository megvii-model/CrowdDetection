import os
import sys, getpass
import os.path as osp
import numpy as np
import pdb
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

root_dir = '../../../..'
add_path(osp.join(root_dir, 'lib'))
add_path(osp.join(root_dir, 'utils'))

class Crowd_human:

    class_names = ['background', 'person']
    num_classes = len(class_names)
    root_folder = '/home/zhenganlin/june/CrowdHuman/'
    image_folder = osp.join(root_folder, 'images')
    train_source = osp.join(root_folder, 'crowd_human_train15000_final_unsure_fixempty_fixvis_vboxmerge.odgt')
    eval_source = osp.join(root_folder, 'crowd_human_test4370_final_unsure_fixempty_fixvis_vboxmerge.odgt')

class Config:

    this_model_dir = osp.split(os.path.realpath(__file__))[0]
    user = getpass.getuser()
    cur_dir = osp.basename(this_model_dir)
    
    output_dir = osp.join(root_dir, 'output', user, 'retinanet', cur_dir)
    model_dir = os.path.join(output_dir, 'model_dump')
    eval_dir = os.path.join(output_dir, 'eval_dump')
    log_dir =  output_dir
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
    datadb = Crowd_human()
    train_source = datadb.train_source
    eval_source = datadb.eval_source
    train_json, eval_json = train_source, eval_source
    image_folder = datadb.image_folder
    imgDir = image_folder
    class_names = datadb.class_names
    num_classes = datadb.num_classes
    class_names2id = dict(list(zip(class_names, list(range(num_classes)))))
    gt_boxes_name = 'fbox'

    # ----------train config---------- #
    backbone_freeze_at = 2
    train_batch_per_gpu = 2
    momentum = 0.9
    weight_decay = 1e-4
    base_lr = 3.125e-4
    learning_rate = base_lr
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2
    anchor_ignore_label = -1

    warm_iter = 1874
    max_epoch = 50
    lr_decay = [0, 30, 40]
    nr_images_epoch = 15000
    log_dump_interval = 1
    iter_per_epoch = nr_images_epoch // train_batch_per_gpu

    # ----------test config---------- #
    test_nms = 0.5
    test_nms_method = 'normal_nms'
    visulize_threshold = 0.3
    pred_cls_threshold = 0.05

    # ----------dataset config---------- #
    nr_box_dim = 5
    max_boxes_of_image = 500

    # --------anchor generator config-------- #
    anchor_base_size = 32 # the minimize anchor size in the bigest feature map.
    anchor_base_scale = [1]
    anchor_aspect_ratios = [1, 2, 3]
    num_cell_anchors = len(anchor_aspect_ratios) * len(anchor_base_scale)

    # ----------binding&training config---------- #
    smooth_l1_beta = 0.1
    negative_thresh = 0.4
    positive_thresh = 0.5
    allow_low_quality = True

config = Config()
