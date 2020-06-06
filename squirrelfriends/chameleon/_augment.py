import albumentations as aug

aug_lst = [
    aug.RandomSunFlare(p=1),
    aug.RandomFog(p=1),
    aug.RandomBrightness(p=1),
    aug.RandomCrop(p=1, height=512, width=512),
    aug.Rotate(p=1, limit=90),
    aug.RGBShift(p=1),
    aug.RandomSnow(p=1),
    aug.HorizontalFlip(p=1),
    aug.VerticalFlip(p=1),
    aug.RandomContrast(limit=0.5, p=1),
    aug.HueSaturationValue(p=1, hue_shift_limit=20,
                           sat_shift_limit=30, val_shift_limit=50)
]
