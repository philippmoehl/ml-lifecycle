import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentations(
        image_size,
        p_augment,
        crop_scale,
        gr_shuffle,
        ssr,
        huesat,
        bricon,
        clahe,
        blur_limit,
        dist_limit,
        cutout,

):
    """
    Get augmentations for training and validation.
    """

    train_augs = A.Compose([
        A.RandomResizedCrop(
            height = image_size, 
            width = image_size,
            scale = crop_scale,
        ),
        A.RandomGridShuffle(
            p = p_augment,
            grid = gr_shuffle,
        ),
        A.Transpose(p = 0.5),
        A.HorizontalFlip(p = 0.5),
        A.VerticalFlip(p   = 0.5),
        A.ShiftScaleRotate(
            p = p_augment,
            shift_limit  = ssr[0],
            scale_limit  = ssr[1],
            rotate_limit = ssr[2],
        ),
        A.HueSaturationValue(
            p = p_augment,
            hue_shift_limit = huesat[0],
            sat_shift_limit = huesat[1],
            val_shift_limit = huesat[2],
        ),
        A.RandomBrightnessContrast(
            p = p_augment,
            brightness_limit = bricon[0],
            contrast_limit   = bricon[1],
        ),
        A.CLAHE(
            p = p_augment,
            clip_limit = clahe[0],
            tile_grid_size = (clahe[1], clahe[1])
        ),
        A.OneOf(
            [
                A.MotionBlur(blur_limit = blur_limit),
                A.MedianBlur(blur_limit = blur_limit),
                A.GaussianBlur(blur_limit = blur_limit)
                ], 
            p = p_augment
        ),
        A.OneOf(
            [
                A.OpticalDistortion(distort_limit = dist_limit),
                A.GridDistortion(distort_limit = dist_limit)
            ], 
            p = p_augment
        ),
        A.Cutout(
            p = p_augment,
            num_holes = cutout[0], 
            max_h_size = int(cutout[1] * image_size),
            max_w_size = int(cutout[1] * image_size)
        ),
        A.Normalize(mean = (0, 0, 0), std = (1, 1, 1)),
        ToTensorV2()
    ])

    test_augs = A.Compose([
        A.SmallestMaxSize(max_size = image_size),
        A.CenterCrop(height = image_size, width  = image_size),
        A.Normalize(mean = (0, 0, 0), std = (1, 1, 1)),
        ToTensorV2()
    ])
    
    # output
    return train_augs, test_augs
