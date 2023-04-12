import os, sys
import tqdm
import argparse

from slide_reader import *
from annotation_reader import Annotation
from config import *
from misc import *



def save_patch_images(annotation_list, patch_info, lesion):

    try:
        for idx, anno in enumerate(annotation_list):
            patch = wsi.get_patch_img(anno, PATCH_SIZE)
            cv2.imwrite(
                os.path.join(
                    save_path, phase, patch_info,
                    "{}_{}_{}.png".format(file_name, lesion, str(idx)),
                ),
                patch,
            )
    except:
        print("except list ", file_name)


def make_patch_images(files, phase):

    if phase == "Train": print("### Training Set")
    elif phase == "val": print("### Validation Set")
    elif phase == "test": print("### Test Set")

    print(len(files))

    for file_name in tqdm.tqdm(files):
        wsi_path = os.path.join(slide_dir, file_name + ".svs")
        anno_mask_path = os.path.join(mask_dir, file_name, info_val[2] + ".png")
        tissue_mask_path = os.path.join(mask_dir, file_name, "tissue.png")

        # WSI & Annotation Instance
        wsi = WSI(
            wsi_path=wsi_path,
            level=1)
        annotation = Annotation(
            wsi,
            anno_mask_path,
            tissue_mask_path,
            isTumor=True, 
            level=1,
            slide_type=info_key)

        # patch_size -> patch_size at level 1
        if phase == "train":
            annotation_list_p, annotation_list_n = annotation.get_annotation(
                patch_size=PATCH_SIZE,
                tumor_ratio_threshold=TUMOR_THRES,
                tissue_ratio_threshold=TISSUE_THRES,
                stride=POS_STRIDE
        )
        elif phase == "val":
            annotation_list_p, annotation_list_n = annotation.get_annotation(
                patch_size=PATCH_SIZE,
                tumor_ratio_threshold=TUMOR_THRES, 
                tissue_ratio_threshold=TISSUE_THRES,
                stride=1
        )
        elif phase == "test":
            annotation_list_p, annotation_list_n = annotation.get_annotation(
                patch_size=PATCH_SIZE,
                tumor_ratio_threshold=TUMOR_THRES, 
                tissue_ratio_threshold=TISSUE_THRES,
                stride=1
        )

#        save_patch_images(annotation_list_p, "p_patch", "pos")
#        save_patch_images(annotation_list_n, "n_patch", "neg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='/home/Dataset/BRCA/')
    parser.add_argument('--subject_case', type=str, default='ovary')
    parser.add_argument('--save_path', type=str, default='./data')

    config = parser.parse_args()

    # Base Path, /home/Dataset/BRCA/{breast or ovary}
    base_path = "{}/{}".format(config.base_path, config.subject_case)

    # Data Output path
    save_path = "{}/{}".format(config.save_path, config.folder)

    info_key = list(SLIDE_TYPE_INFO.keys())[0]
    info_val = list(SLIDE_TYPE_INFO.values())[0]

    slide_dir = os.path.join(base_path, info_val[0])
    mask_dir = os.path.join(base_path, info_val[1])

    files = os.listdir(mask_dir)
    fake_label = np.empty([len(files)])

    files_train, files_val, files_test = file_split(
        files,
        fake_label,
        Train_Ratio,
        int(config.folder)
    )

    print(len(files_train), len(files_val), len(files_test))

    make_patch_images(files_test, "test")
    make_patch_images(files_val, "validation")
    make_patch_images(files_train, "train")
