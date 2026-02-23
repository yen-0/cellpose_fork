"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu and Michael Rariden.
"""

import argparse


def get_arg_parser():
    """ Parses command line arguments for cellpose main function

    Note: this function has to be in a separate file to allow autodoc to work for CLI.
    The autodoc_mock_imports in conf.py does not work for sphinx-argparse sometimes,
    see https://github.com/ashb/sphinx-argparse/issues/9#issue-1097057823
    """

    parser = argparse.ArgumentParser(description="Cellpose Command Line Parameters")

    # misc settings
    parser.add_argument("--version", action="store_true",
                        help="show cellpose version info")
    parser.add_argument(
        "--verbose", action="store_true",
        help="show information about running and settings and save to log")
    parser.add_argument("--Zstack", action="store_true", help="run GUI in 3D mode")

    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("Hardware Arguments")
    hardware_args.add_argument("--use_gpu", action="store_true",
                               help="use gpu if torch with cuda installed")
    hardware_args.add_argument(
        "--gpu_device", required=False, default="0", type=str,
        help="which gpu device to use, use an integer for torch, or mps for M1")
    
    # settings for locating and formatting images
    input_img_args = parser.add_argument_group("Input Image Arguments")
    input_img_args.add_argument("--dir", default=[], type=str,
                                help="folder containing data to run or train on.")
    input_img_args.add_argument(
        "--image_path", default=[], type=str, help=
        "if given and --dir not given, run on single image instead of folder (cannot train with this option)"
    )
    input_img_args.add_argument(
        "--look_one_level_down", action="store_true",
        help="run processing on all subdirectories of current folder")
    input_img_args.add_argument("--img_filter", default=[], type=str,
                                help="end string for images to run on")
    input_img_args.add_argument(
        "--channel_axis", default=None, type=int,
        help="axis of image which corresponds to image channels")
    input_img_args.add_argument("--z_axis", default=None, type=int,
                                help="axis of image which corresponds to Z dimension")
    
    # TODO: remove deprecated in future version
    input_img_args.add_argument(
        "--chan", default=0, type=int, help=
        "Deprecated in v4.0.1+, not used. ")
    input_img_args.add_argument(
        "--chan2", default=0, type=int, help=
        'Deprecated in v4.0.1+, not used. ')
    input_img_args.add_argument("--invert", action="store_true", help=
        'Deprecated in v4.0.1+, not used. ')
    input_img_args.add_argument(
        "--all_channels", action="store_true", help=
        'Deprecated in v4.0.1+, not used. ')

    # model settings
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument("--pretrained_model", required=False, default="cpsam",
                            type=str,
                            help="model to use for running or starting training")
    model_args.add_argument(
        "--add_model", required=False, default=None, type=str,
        help="model path to copy model to hidden .cellpose folder for using in GUI/CLI")
    model_args.add_argument("--pretrained_model_ortho", required=False, default=None,
                            type=str,
                            help="Deprecated in v4.0.1+, not used. ")
    
    # TODO: remove deprecated in future version
    model_args.add_argument("--restore_type", required=False, default=None, type=str, help=
        'Deprecated in v4.0.1+, not used. ')
    model_args.add_argument("--chan2_restore", action="store_true", help=
        'Deprecated in v4.0.1+, not used. ')
    model_args.add_argument(
        "--transformer", action="store_true", help=
        "use transformer backbone (pretrained_model from Cellpose3 is transformer_cp3)")
    
    # algorithm settings
    algorithm_args = parser.add_argument_group("Algorithm Arguments")
    algorithm_args.add_argument("--no_norm", action="store_true",
                                help="do not normalize images (normalize=False)")
    algorithm_args.add_argument(
        '--norm_percentile',
        nargs=2,  # Require exactly two values
        metavar=('VALUE1', 'VALUE2'),
        help="Provide two float values to set norm_percentile (e.g., --norm_percentile 1 99)"
    )
    algorithm_args.add_argument(
        "--do_3D", action="store_true",
        help="process images as 3D stacks of images (nplanes x nchan x Ly x Lx")
    algorithm_args.add_argument(
        "--diameter", required=False, default=None, type=float, help=
        "use to resize cells to the training diameter (30 pixels)"
    )
    algorithm_args.add_argument(
        "--stitch_threshold", required=False, default=0.0, type=float,
        help="compute masks in 2D then stitch together masks with IoU>0.9 across planes"
    )
    algorithm_args.add_argument(
        "--min_size", required=False, default=15, type=int,
        help="minimum number of pixels per mask, can turn off with -1")
    algorithm_args.add_argument(
        "--flow3D_smooth", required=False, default=0, type=float,
        help="stddev of gaussian for smoothing of dP for dynamics in 3D, default of 0 means no smoothing")
    algorithm_args.add_argument(
        "--flow_threshold", default=0.4, type=float, help=
        "flow error threshold, 0 turns off this optional QC step. Default: %(default)s")
    algorithm_args.add_argument(
        "--cellprob_threshold", default=0, type=float,
        help="cellprob threshold, default is 0, decrease to find more and larger masks")
    algorithm_args.add_argument(
        "--niter", default=0, type=int, help=
        "niter, number of iterations for dynamics for mask creation, default of 0 means it is proportional to diameter, set to a larger number like 2000 for very long ROIs"
    )
    algorithm_args.add_argument("--anisotropy", required=False, default=1.0, type=float,
                                help="anisotropy of volume in 3D")
    algorithm_args.add_argument("--exclude_on_edges", action="store_true",
                                help="discard masks which touch edges of image")
    algorithm_args.add_argument(
        "--augment", action="store_true",
        help="tiles image with overlapping tiles and flips overlapped regions to augment"
    )
    algorithm_args.add_argument("--batch_size", default=8, type=int,
                               help="inference batch size. Default: %(default)s")

    # TODO: remove deprecated in future version
    algorithm_args.add_argument(
        "--no_resample", action="store_true", 
        help="disables flows/cellprob resampling to original image size before computing masks. Using this flag will make more masks more jagged with larger diameter settings.")
    algorithm_args.add_argument(
        "--no_interp", action="store_true",
        help="do not interpolate when running dynamics (was default)")

    # output settings
    output_args = parser.add_argument_group("Output Arguments")
    output_args.add_argument(
        "--save_png", action="store_true",
        help="save masks as png")
    output_args.add_argument(
        "--save_tif", action="store_true",
        help="save masks as tif")
    output_args.add_argument(
        "--output_name", default=None, type=str,
        help="suffix for saved masks, default is _cp_masks, can be empty if `savedir` used and different of `dir`")
    output_args.add_argument("--no_npy", action="store_true",
                             help="suppress saving of npy")
    output_args.add_argument(
        "--savedir", default=None, type=str, help=
        "folder to which segmentation results will be saved (defaults to input image directory)"
    )
    output_args.add_argument(
        "--dir_above", action="store_true", help=
        "save output folders adjacent to image folder instead of inside it (off by default)"
    )
    output_args.add_argument("--in_folders", action="store_true",
                             help="flag to save output in folders (off by default)")
    output_args.add_argument(
        "--save_flows", action="store_true", help=
        "whether or not to save RGB images of flows when masks are saved (disabled by default)"
    )
    output_args.add_argument(
        "--save_outlines", action="store_true", help=
        "whether or not to save RGB outline images when masks are saved (disabled by default)"
    )
    output_args.add_argument(
        "--save_rois", action="store_true",
        help="whether or not to save ImageJ compatible ROI archive (disabled by default)"
    )
    output_args.add_argument(
        "--save_txt", action="store_true",
        help="flag to enable txt outlines for ImageJ (disabled by default)")
    output_args.add_argument(
        "--save_mpl", action="store_true",
        help="save a figure of image/mask/flows using matplotlib (disabled by default). "
        "This is slow, especially with large images.")

    # training settings
    training_args = parser.add_argument_group("Training Arguments")
    training_args.add_argument("--train", action="store_true",
                               help="train network using images in dir")
    training_args.add_argument("--test_dir", default=[], type=str,
                               help="folder containing test data (optional)")
    training_args.add_argument(
        "--file_list", default=[], type=str, help=
        "path to list of files for training and testing and probabilities for each image (optional)"
    )
    training_args.add_argument(
        "--mask_filter", default="_masks", type=str, help=
        "end string for masks to run on. use '_seg.npy' for manual annotations from the GUI. Default: %(default)s"
    )
    training_args.add_argument("--learning_rate", default=1e-5, type=float,
                               help="learning rate. Default: %(default)s")
    training_args.add_argument("--weight_decay", default=0.1, type=float,
                               help="weight decay. Default: %(default)s")
    training_args.add_argument("--n_epochs", default=100, type=int,
                               help="number of epochs. Default: %(default)s")
    training_args.add_argument("--train_batch_size", default=1, type=int,
                               help="training batch size. Default: %(default)s")
    training_args.add_argument("--bsize", default=256, type=int,
                               help="block size for tiles. Default: %(default)s")
    training_args.add_argument(
        "--nimg_per_epoch", default=None, type=int,
        help="number of train images per epoch. Default is to use all train images.")
    training_args.add_argument(
        "--nimg_test_per_epoch", default=None, type=int,
        help="number of test images per epoch. Default is to use all test images.")
    training_args.add_argument(
        "--min_train_masks", default=5, type=int, help=
        "minimum number of masks a training image must have to be used. Default: %(default)s"
    )
    training_args.add_argument("--SGD", default=0, type=int, 
                               help="Deprecated in v4.0.1+, not used - AdamW used instead. ")
    training_args.add_argument(
        "--save_every", default=100, type=int,
        help="number of epochs to skip between saves. Default: %(default)s")
    training_args.add_argument(
        "--save_each", action="store_true",
        help="wether or not to save each epoch. Must also use --save_every. (default: False)")
    training_args.add_argument(
        "--model_name_out", default=None, type=str,
        help="Name of model to save as, defaults to name describing model architecture. "
        "Model is saved in the folder specified by --dir in models subfolder.")
    
    # TODO: remove deprecated in future version
    training_args.add_argument(
        "--diam_mean", default=30., type=float, help=
        'Deprecated in v4.0.1+, not used. ')
    training_args.add_argument("--train_size", action="store_true", help=
        'Deprecated in v4.0.1+, not used. ')

    # semi3d settings
    semi3d_args = parser.add_argument_group("Semi3D Arguments")
    semi3d_args.add_argument("--semi3d", action="store_true",
                             help="run semi-3D inference (2D slices + z-linking/reconstruction)")
    semi3d_args.add_argument("--semi3d_train", action="store_true",
                             help="train semi-3D refinement classifier")
    semi3d_args.add_argument("--semi3d_stage1", action="store_true",
                             help="run semi-3D stage1 only (2D per-slice inference + save masks)")
    semi3d_args.add_argument("--semi3d_stage2", action="store_true",
                             help="run semi-3D stage2 only (linking/reconstruction from saved stage1 masks)")
    semi3d_args.add_argument("--semi3d_eval", action="store_true",
                             help="evaluate semi-3D outputs")
    semi3d_args.add_argument("--semi3d_input", default=None, type=str,
                             help="input stack path or stack directory for semi3d/semi3d_train")
    semi3d_args.add_argument("--semi3d_output", default=None, type=str,
                             help="output directory for semi3d modes")
    semi3d_args.add_argument("--semi3d_pred", default=None, type=str,
                             help="predicted stack path for semi3d_eval")
    semi3d_args.add_argument("--semi3d_gt", default=None, type=str,
                             help="ground-truth stack path for semi3d_eval")
    semi3d_args.add_argument("--semi3d_pretrained_model", default="cpsam", type=str,
                             help="pretrained model for semi3d inference/training")
    semi3d_args.add_argument("--semi3d_diameter", default=None, type=float,
                             help="diameter used during per-slice 2D inference")
    semi3d_args.add_argument("--semi3d_seed", default=0, type=int,
                             help="deterministic seed for semi3d modes")
    semi3d_args.add_argument("--semi3d_link_iou", default=0.1, type=float,
                             help="minimum overlap for adjacent linking")
    semi3d_args.add_argument("--semi3d_link_dist", default=30.0, type=float,
                             help="max centroid distance for linking")
    semi3d_args.add_argument("--semi3d_size_tolerance", default=0.6, type=float,
                             help="minimum relative area similarity for linking")
    semi3d_args.add_argument("--semi3d_max_gap", default=3, type=int,
                             help="max z gap allowed in track linking")
    semi3d_args.add_argument("--semi3d_min_track_len", default=2, type=int,
                             help="minimum linked length to keep a track")
    semi3d_args.add_argument("--semi3d_min_conf", default=0.05, type=float,
                             help="minimum final track confidence")
    semi3d_args.add_argument("--semi3d_save_3d_labels", action="store_true",
                             help="save 3D z-consistent track labels for semi3d inference")
    semi3d_args.add_argument("--semi3d_refiner_model", default=None, type=str,
                             help="path to trained semi3d_refiner.npz for track filtering during inference")
    semi3d_args.add_argument("--semi3d_stage1_masks", default=None, type=str,
                             help="path to semi3d_stage1_masks.tif (or .npy) for --semi3d_stage2")
    semi3d_args.add_argument("--semi3d_stage1_flows", default=None, type=str,
                             help="optional path to semi3d_stage1_flows.npy for --semi3d_stage2")
    semi3d_args.add_argument("--semi3d_stage1_prob", default=None, type=str,
                             help="optional path to semi3d_stage1_prob.tif for --semi3d_stage2")
    semi3d_args.add_argument("--semi3d_border_exclusion_px", default=0, type=int,
                             help="remove stage1 instances touching border band of this width")
    semi3d_args.add_argument("--semi3d_fill_edges", action="store_true",
                             help="fill reconstructed track masks on first/last stack slices")
    semi3d_args.add_argument("--semi3d_save_flows", action="store_true",
                             help="save stage1 flow hints for stage2 reconstruction")
    semi3d_args.add_argument("--semi3d_save_debug_tiff", action="store_true",
                             help="save debug maps (flow magnitude / probability / refiner keep prob) as TIFF")
    semi3d_args.add_argument("--semi3d_stage2_use_gpu", action="store_true",
                             help="use GPU acceleration for stage2 refiner scoring when available")
    semi3d_args.add_argument("--semi3d_memmap_stage2_inputs", action="store_true",
                             help="memory-map stage2 input masks when using .npy inputs to reduce RAM usage")
    semi3d_args.add_argument("--semi3d_link_gpu_prefilter", action="store_true",
                             help="use hybrid GPU prefilter (distance/size gating) during stage2 linking")
    semi3d_args.add_argument("--semi3d_merge_dist", default=12.0, type=float,
                             help="max centroid distance to merge split detections within a slice")
    semi3d_args.add_argument("--semi3d_allow_overlap_recon", action="store_true",
                             help="allow reconstructed masks to overlap already-established masks")
    semi3d_args.add_argument("--semi3d_recon_min_free_fraction", default=0.25, type=float,
                             help="minimum free-pixel fraction required to keep reconstructed mask")
    semi3d_args.add_argument("--semi3d_use_prob_occupancy", action="store_true",
                             help="use stage1 probability map as additional occupancy prior in stage2 reconstruction")
    semi3d_args.add_argument("--semi3d_prob_occupancy_thresh", default=0.5, type=float,
                             help="probability threshold above which territories are treated as occupied for reconstruction")
    semi3d_args.add_argument("--semi3d_disable_stage1_prob_defog", action="store_true",
                             help="disable stage1 probability de-fogging before saving stage1 prob TIFF")
    semi3d_args.add_argument("--semi3d_stage1_prob_bg_percentile", default=5.0, type=float,
                             help="background percentile after stage1 fog-field subtraction")
    semi3d_args.add_argument("--semi3d_stage1_prob_hi_percentile", default=97.0, type=float,
                             help="high percentile after stage1 fog-field subtraction")
    semi3d_args.add_argument("--semi3d_stage1_prob_gamma", default=0.9, type=float,
                             help="gamma after stage1 fog-field subtraction and percentile normalization")
    semi3d_args.add_argument("--semi3d_stage1_prob_bg_sigma", default=40.0, type=float,
                             help="gaussian sigma (pixels) for spatial fog-field estimation in stage1")
    semi3d_args.add_argument("--semi3d_stage1_cellprob_threshold", default=-2.0, type=float,
                             help="cellprob threshold for stage1 2D Cellpose eval (lower => keep more candidate structures)")
    semi3d_args.add_argument("--semi3d_refiner_threshold", default=0.5, type=float,
                             help="keep threshold for trained semi3d refiner probability")
    semi3d_args.add_argument("--semi3d_simulate_z_dropout", action="store_true",
                             help="simulate z-dropout when training semi3d refiner")
    semi3d_args.add_argument("--semi3d_dropout_prob", default=0.2, type=float,
                             help="dropout probability for z-dropout simulation")
    semi3d_args.add_argument("--semi3d_learning_rate", default=1e-2, type=float,
                             help="learning rate for semi3d refiner")
    semi3d_args.add_argument("--semi3d_epochs", default=200, type=int,
                             help="training epochs for semi3d refiner")
    semi3d_args.add_argument("--semi3d_refiner_linear", action="store_true",
                             help="use linear refiner instead of nonlinear model")
    semi3d_args.add_argument("--semi3d_refiner_hidden_dim", default=16, type=int,
                             help="hidden dimension for nonlinear refiner")
    semi3d_args.add_argument("--semi3d_refiner_batch_size", default=64, type=int,
                             help="mini-batch size for refiner training")
    semi3d_args.add_argument("--semi3d_synthetic_per_obj", default=6, type=int,
                             help="number of synthetic distorted samples per GT object for refiner training")


    return parser
