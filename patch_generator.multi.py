import os
import time

from multiprocessing import Process, Queue

from factory.slide_reader import *
from factory.annotation_reader import Annotation
from factory.config import *
from factory.misc import *

import argparse

def split_sublist(file_list, num_files):
    return [ file_list[ i * num_files: (i + 1) * num_files ] for i in range((len(file_list) + num_files - 1) // num_files) ]

def patch_generator(files, phase):
    for file_name in files:
        wsi_path = os.path.join(slide_dir, file_name + ".svs")
        anno_mask_path = os.path.join(mask_dir, file_name, info_val[2] + ".png")
        tissue_mask_path = os.path.join(mask_dir, file_name, "tissue.png")

        if not os.path.exists(wsi_path):
            continue

        # WSI & Annotation Instance
        wsi = WSI(wsi_path=wsi_path, level=1)
        annotation = Annotation(wsi, anno_mask_path, tissue_mask_path, isTumor=True, level=1, slide_type=info_key)

        # patch_size -> patch_size at level 1
        if phase == "train":
            annotation_list_p, annotation_list_n, hw_list_p, hw_list_n = annotation.get_annotation(
                patch_size=IMG_SIZE, stride=POS_STRIDE,
                tumor_ratio_threshold=TUMOR_THRES, tissue_ratio_threshold=TISSUE_THRES,
        )
        elif phase == "val":
            annotation_list_p, annotation_list_n, hw_list_p, hw_list_n = annotation.get_annotation(
                patch_size=IMG_SIZE, stride=1,
                tumor_ratio_threshold=TUMOR_THRES, tissue_ratio_threshold=TISSUE_THRES,
        )
        elif phase == "test":
            with open(os.path.join(save_path, "test.txt"), "a") as f:
                f.write(file_name + "\n") 
            annotation_list_p, annotation_list_n, hw_list_p, hw_list_n = annotation.get_annotation(
                patch_size=IMG_SIZE, stride=1,
                tumor_ratio_threshold=TUMOR_THRES, tissue_ratio_threshold=TISSUE_THRES,
        )

        for idx, anno in enumerate(annotation_list_p):
            patch = wsi.get_patch_img(anno, IMG_SIZE) 

            cv2.imwrite("{}/{}/p_patch/{}_pos_h{}_w{}.png".format(save_path, phase, file_name, hw_list_p[idx][0], hw_list_p[idx][1]), patch)
            
        for idx, anno in enumerate(annotation_list_n):
            patch = wsi.get_patch_img(anno, IMG_SIZE)
            cv2.imwrite("{}/{}/n_patch/{}_neg_h{}_w{}.png".format(save_path, phase, file_name, hw_list_n[idx][0], hw_list_n[idx][1]), patch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=int, default=0, help='(default=0)')
    parser.add_argument('--base_path', type=str, default='/home/Dataset/BRCA/', help='/home/Dataset/BRCA/')
    parser.add_argument('--subject_case', type=str, default='ovary', help='ovary or breast')
    parser.add_argument('--save_path', type=str, default='./data', help='./data')
    parser.add_argument('--annotation', type=str, default='label', help='(label or unlabel)')

    config = parser.parse_args()
    
    # Base Path, /home/Dataset/BRCA/{breast or ovary}
    base_path = "{}/{}".format(config.base_path, config.subject_case)
    # Data Output path
    save_path = "{}/{}".format(config.save_path, config.folder)
    # factory.config: SLIDE_TYPE_INFO
    if config.annotation == 'label':
        info_key = list(SLIDE_TYPE_INFO_LABEL.keys())[0]
        info_val = list(SLIDE_TYPE_INFO_LABEL.values())[0]
    elif config.annotation == 'unlabel':
        info_key = list(SLIDE_TYPE_INFO_UNLABEL.keys())[0]
        info_val = list(SLIDE_TYPE_INFO_UNLABEL.values())[0]
    
    p_patch_train = os.path.join(save_path, "train", "p_patch")
    n_patch_train = os.path.join(save_path, "train", "n_patch")
    p_patch_val = os.path.join(save_path, "val", "p_patch")
    n_patch_val = os.path.join(save_path, "val", "n_patch")
    p_patch_test = os.path.join(save_path, "test", "p_patch")
    n_patch_test = os.path.join(save_path, "test", "n_patch")

    if not os.path.exists(p_patch_train): os.makedirs(p_patch_train)
    if not os.path.exists(n_patch_train): os.makedirs(n_patch_train)
    if not os.path.exists(p_patch_val): os.makedirs(p_patch_val)
    if not os.path.exists(n_patch_val): os.makedirs(n_patch_val)
    if not os.path.exists(p_patch_test): os.makedirs(p_patch_test)
    if not os.path.exists(n_patch_test): os.makedirs(n_patch_test)

    slide_dir = os.path.join(base_path, info_val[0])
    mask_dir = os.path.join(base_path, info_val[1])

    files = os.listdir(mask_dir)
    fake_label = np.empty([len(files)])

    procs = []
    n = 4

    if config.annotation == 'label':
        files_train, files_val, files_test = file_split(files, fake_label, Train_Ratio)
    
        for idx, sub_list_train in enumerate(split_sublist(files_train, n)):
            print("## {} ## Training Set {} - {}".format(time.strftime('%c', time.localtime(time.time())), idx, sub_list_train))
            proc = Process(target=patch_generator, args=(sub_list_train, 'train'))
            procs.append(proc)
            proc.start()
        
        for idx, sub_list_val in enumerate(split_sublist(files_val, n)):
            print("## {} ## Validation Set {} - {}".format(time.strftime('%c', time.localtime(time.time())), idx, sub_list_val))
            proc = Process(target=patch_generator, args=(sub_list_val, 'val'))
            procs.append(proc)
            proc.start()
    
        for idx, sub_list_test in enumerate(split_sublist(files_test, n)):
            print("## {} ## Test Set {} - {}".format(time.strftime('%c', time.localtime(time.time())), idx, sub_list_test))
            proc = Process(target=patch_generator, args=(sub_list_test, 'test'))
            procs.append(proc)
            proc.start()
    elif config.annotation == 'unlabel':
        files_test = files

        for idx, sub_list_test in enumerate(split_sublist(files_test, n)):
            print("## {} ## Test Set {} - {}".format(time.strftime('%c', time.localtime(time.time())), idx, sub_list_test))
            proc = Process(target=patch_generator, args=(sub_list_test, 'test'))
            procs.append(proc)
            proc.start()

    # w/o join method, process will stay idle state - zombie process
    for proc in procs:
        proc.join()

    print('## {} ## All Process of Patch-Generation End'.format(time.strftime('%c', time.localtime(time.time()))))
