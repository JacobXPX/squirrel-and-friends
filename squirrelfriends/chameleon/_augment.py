import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(in_size=[1024, 1024],
                   out_size=[512, 512],
                   prob=0.5,
                   add_cutout=True,
                   to_tensorv2=True,
                   bbox_params=None,
                   keypoint_params=None,
                   additional_targets=None):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        prob (float): probability of applying all list of transforms. Default: 0.5
        add_cutout (boolean): if adding cutout on image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}.

    Returns:
        aug_compose (Compose): common Compose transforms for train images
    """

    phase_transforms = {}
    phase_transforms["train"] = get_train_transforms(
        in_size=in_size,
        out_size=out_size,
        prob=prob,
        add_cutout=add_cutout,
        to_tensorv2=to_tensorv2,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets
    )
    phase_transforms["valid"] = get_train_transforms(
        in_size=in_size,
        out_size=out_size,
        to_tensorv2=to_tensorv2,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets
    )
    phase_transforms["test"] = get_train_transforms(
        in_size=in_size,
        out_size=out_size,
        to_tensorv2=to_tensorv2
    )
    return phase_transforms


def get_train_transforms(in_size=[1024, 1024],
                         out_size=[512, 512],
                         prob=0.5,
                         add_cutout=True,
                         to_tensorv2=True,
                         bbox_params=None,
                         keypoint_params=None,
                         additional_targets=None
                         ):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        prob (float): probability of applying all list of transforms. Default: 0.5
        add_cutout (boolean): if adding cutout on image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}.

    Returns:
        aug_compose (Compose): common Compose transforms for train images
    """

    compose_lst = [
        aug.RandomSizedCrop(
            min_max_height=(
                int(in_size[1] * 0.7), int(in_size[1] * 1.0)),
            height=in_size[0],
            width=in_size[1],
            p=prob
        ),
        aug.OneOf([
            aug.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=prob
            ),
            aug.RandomContrast(
                limit=0.2,
                p=prob
            ),
            aug.RGBShift(
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=prob
            ),
            aug.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=prob
            ),
        ],
            p=prob
        ),
        aug.OneOf([
            aug.HorizontalFlip(p=prob),
            aug.VerticalFlip(p=prob),
            aug.RandomRotate90(p=prob)
        ],
            p=prob
        ),
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        )
    ]
    if add_cutout:
        compose_lst.append(
            aug.Cutout(
                num_holes=8,
                max_h_size=int(out_size//16),
                max_w_size=int(out_size//16),
                fill_value=0,
                p=prob
            )
        )
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets,
    )

    return aug_compose


def get_valid_transforms(in_size=[1024, 1024],
                         out_size=[512, 512],
                         to_tensorv2=True,
                         bbox_params=None,
                         keypoint_params=None,
                         additional_targets=None
                         ):
    """
    Get a common Compose transforms for training images.
    For bbox_params, keypoint_params, additional_targets arguments,
    Refering https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.
        bbox_params (BboxParams): Parameters for bounding boxes transforms.
        keypoint_params (KeypointParams): Parameters for keypoints transforms.
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}.

    Returns:
        aug_compose (Compose): common Compose transforms for valid images
    """

    compose_lst = [
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        ),
    ]
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0,
        bbox_params=bbox_params,
        keypoint_params=keypoint_params,
        additional_targets=additional_targets,
    )

    return aug_compose


def get_test_transforms(in_size=[1024, 1024],
                        out_size=[512, 512],
                        to_tensorv2=True
                        ):
    """
    Get a common Compose transforms for training images.

    Args:
        in_size (int, int]): size of input image.
        out_size ([int, int]): size of output image.
        to_tensorv2 (boolean): if converting it to pytorch tensor.

    Returns:
        aug_compose (Compose): common Compose transforms for test images
    """

    compose_lst = [
        aug.Resize(
            height=out_size[0],
            width=out_size[1],
            p=1.0
        ),
    ]
    if to_tensorv2:
        compose_lst.append(ToTensorV2(p=1.0))

    aug_compose = aug.Compose(
        compose_lst,
        p=1.0
    )

    return aug_compose
