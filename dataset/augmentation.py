import random
import albumentations as A

random.seed(42)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

default_transform = A.Compose(
    [
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pad_if_needed=True),
        A.ToTensorV2(transpose_mask=True),
    ],
    seed=137,
    strict=True,
)

augmented_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # A.CenterCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, pad_if_needed=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-40, 40)),
        A.ElasticTransform(p=0.5),
        A.ColorJitter(),
        A.ToTensorV2(transpose_mask=True),
    ],
    seed=137,
    strict=True,
)