IMG_SIZE = 512
TUMOR_THRES = 50 # folder 3
TISSUE_THRES = 80 # folder 3

POS_STRIDE = 3
NEG_STRIDE = 1
Train_Ratio = 0.80

# phenotype = [slide_dir, mask_dir, annotation(lesion), subject size]
SLIDE_TYPE_INFO_LABEL = {
    "positive": ["raw_images_label", "mask_images_label", "Lesion", 1]
}

SLIDE_TYPE_INFO_UNLABEL = {
    "positive": ["raw_images_unlabel", "mask_images_unlabel", "Lesion", 1]
}

model_name_list = [
     'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
     'vit', 'cait', 'deepvit',
     'resnet50', 'resnet101', 'resnet152',
     'densenet121', 'densenet161', 'densenet169', 'densenet201',
     'inception_v3'
]
