"""Dataset paths (machine-specific)."""

IMAGENET_DIR = "/temp/pemb6612/back_up/pemb6612/exploration/SSL/data"
IMAGENET_TRAIN_TAR = IMAGENET_DIR + "/ILSVRC2012_img_train.tar"   # 150GB, not extracted
IMAGENET_VAL_TAR = IMAGENET_DIR + "/ILSVRC2012_img_val.tar"        # 6.9GB
IMAGENET_VAL_DIR = IMAGENET_DIR + "/imagenet100/val"              # 50k .JPEG extracted (flat)
IMAGENET_DEVKIT = IMAGENET_DIR + "/devkit_t12/ILSVRC2012_devkit_t12"

# Extracted ImageFolder layout (local NVMe, fast) — populated by
# extract_imagenet.sh: train/<wnid>/*.JPEG and val/<wnid>/*.JPEG
IMAGENET_EXTRACTED = "/temp/pemb6612/imagenet"
IMAGENET_TRAIN = IMAGENET_EXTRACTED + "/train"
IMAGENET_VAL = IMAGENET_EXTRACTED + "/val"
