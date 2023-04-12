import os
import cv2
import random
import numpy as np


class Annotation:
    def __init__(self, wsi, anno_mask_path, tissue_mask_path, isTumor=True, level=1, slide_type="positive"):
        self.level = level
        self.wsi = wsi
        self.slide_type = slide_type

        self.anno_mask = self.get_mask_img(anno_mask_path)
        self.tissue_mask = self.get_mask_img(tissue_mask_path)
#        self.mask_size_ratio = int(self.wsi.height * pow(2, self.level) / self.tissue_mask.shape[0])
        self.mask_size_ratio = 32

    def get_mask_img(self, mask_path):
        if os.path.isfile(mask_path):
            mask_img = cv2.imread(mask_path)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        else:
            mask_img = np.zeros([100, 100], np.uint8)
        return mask_img

    def get_annotation(self, patch_size, tumor_ratio_threshold=80, tissue_ratio_threshold=80, stride=2):
        b_anno = cv2.threshold(self.anno_mask, 127, 255, cv2.THRESH_BINARY)[1]
        b_tissue = cv2.threshold(self.tissue_mask, 127, 255, cv2.THRESH_BINARY)[1]
        height, width = b_tissue.shape
        pos_annotation_list = []
        neg_annotation_list = []
        pos_hw_list = []
        neg_hw_list = []
        
        resized_patch_size = int(patch_size * pow(2, self.level) / self.mask_size_ratio)

        h_start = np.arange(0, height, int(resized_patch_size / stride))
        w_start = np.arange(0, width, int(resized_patch_size / stride))

        for h, h_val in enumerate(h_start):
            for w, w_val in enumerate(w_start):

                anno_area = b_anno[
                    h_val : h_val + resized_patch_size,
                    w_val : w_val + resized_patch_size,
                ]
                tissue_area = b_tissue[
                    h_val : h_val + resized_patch_size,
                    w_val : w_val + resized_patch_size,
                ]

                tumor_ratio = np.sum(anno_area) / (resized_patch_size * resized_patch_size)
                tumor_ratio = int( ( tumor_ratio / 255 ) * 100 )
                tissue_ratio = np.sum(tissue_area) / (resized_patch_size * resized_patch_size)
                tissue_ratio = int( ( tissue_ratio / 255 ) * 100 )


                tumor_ratio_threshold_min = 20
                tissue_ratio_threshold_min = 50

                if ( tumor_ratio >= tumor_ratio_threshold and tissue_ratio > tissue_ratio_threshold ):
                    pos_annotation_list.append( (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,) )
                    pos_hw_list.append([h_val, w_val]) 
                elif ( tumor_ratio < tumor_ratio_threshold_min and tissue_ratio > tissue_ratio_threshold ):
                    neg_annotation_list.append( (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,) )
                    neg_hw_list.append([h_val, w_val])
                elif ( tumor_ratio >= tumor_ratio_threshold_min and tumor_ratio < tumor_ratio_threshold and tissue_ratio > tissue_ratio_threshold ):
                    continue
                    

        return pos_annotation_list, neg_annotation_list, pos_hw_list, neg_hw_list
    
    def get_annotation_test_positive(self, patch_size, tumor_ratio_threshold=80, tissue_ratio_threshold=80, stride=2):
        b_anno = cv2.threshold(self.anno_mask, 127, 255, cv2.THRESH_BINARY)[1]
        b_tissue = cv2.threshold(self.tissue_mask, 127, 255, cv2.THRESH_BINARY)[1]
        height, width = b_tissue.shape
        pos_annotation_list = []
        neg_annotation_list = []
        resized_patch_size = int(patch_size * pow(2, self.level) / self.mask_size_ratio)

        h_start = np.arange(0, height, int(resized_patch_size / stride))
        w_start = np.arange(0, width, int(resized_patch_size / stride))

        for h, h_val in enumerate(h_start):
            for w, w_val in enumerate(w_start):

                anno_area = b_anno[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tissue_area = b_tissue[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tumor_ratio = np.sum(anno_area) / (resized_patch_size * resized_patch_size)
                tumor_ratio = int(tumor_ratio * 100 / 255)
                tissue_ratio = np.sum(tissue_area) / (resized_patch_size * resized_patch_size)
                tissue_ratio = int(tissue_ratio * 100 / 255)
                
                if (
                    (tumor_ratio > tumor_ratio_threshold)
                    and (tissue_ratio > tissue_ratio_threshold)
                ):
                    pos_annotation_list.append(
                        (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,)
                    )

                elif (
                    (tumor_ratio < tumor_ratio_threshold)
                    and (tissue_ratio > tissue_ratio_threshold)
                ):                    # Here needs foreground check
                    neg_annotation_list.append(
                        (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,)
                    )
        
        return pos_annotation_list, neg_annotation_list
    
    def get_annotation_test_positive(self, patch_size, tumor_ratio_threshold=80, tissue_ratio_threshold=80, stride=2):
        b_anno = cv2.threshold(self.anno_mask, 127, 255, cv2.THRESH_BINARY)[1]
        b_tissue = cv2.threshold(self.tissue_mask, 127, 255, cv2.THRESH_BINARY)[1]
        height, width = b_tissue.shape
        pos_annotation_list = []
        neg_annotation_list = []
        resized_patch_size = int(patch_size * pow(2, self.level) / self.mask_size_ratio)

        h_start = np.arange(0, height, int(resized_patch_size / stride))
        w_start = np.arange(0, width, int(resized_patch_size / stride))

        for h, h_val in enumerate(h_start):
            for w, w_val in enumerate(w_start):

                anno_area = b_anno[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tissue_area = b_tissue[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tumor_ratio = np.sum(anno_area) / (resized_patch_size * resized_patch_size)
                tumor_ratio = int(tumor_ratio * 100 / 255)
                tissue_ratio = np.sum(tissue_area) / (resized_patch_size * resized_patch_size)
                tissue_ratio = int(tissue_ratio * 100 / 255)

                if ( tumor_ratio > tumor_ratio_threshold and tissue_ratio > tissue_ratio_threshold ):
                    pos_annotation_list.append( (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,) )
                elif ( tumor_ratio < tumor_ratio_threshold and tissue_ratio > tissue_ratio_threshold ):
                    # Here needs foreground check
                    neg_annotation_list.append( (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,) )

                return pos_annotation_list, neg_annotation_list
                
    def get_annotation_test_negative(self, patch_size, tumor_ratio_threshold=80, tissue_ratio_threshold=80, stride=2):
        b_anno = cv2.threshold(self.anno_mask, 127, 255, cv2.THRESH_BINARY)[1]
        b_tissue = cv2.threshold(self.tissue_mask, 127, 255, cv2.THRESH_BINARY)[1]
        height, width = b_tissue.shape
        annotation_list = []
        resized_patch_size = int(patch_size * pow(2, self.level) / self.mask_size_ratio)

        h_start = np.arange(0, height, int(resized_patch_size / stride))
        w_start = np.arange(0, width, int(resized_patch_size / stride))

        for h, h_val in enumerate(h_start):
            for w, w_val in enumerate(w_start):

                anno_area = b_anno[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tissue_area = b_tissue[
                    h_val : h_val + resized_patch_size, w_val : w_val + resized_patch_size,
                ]
                tumor_ratio = np.sum(anno_area) / (resized_patch_size * resized_patch_size)
                tumor_ratio = int(tumor_ratio * 100 / 255)
                tissue_ratio = np.sum(tissue_area) / (resized_patch_size * resized_patch_size)
                tissue_ratio = int(tissue_ratio * 100 / 255)
                

                if tissue_ratio > tissue_ratio_threshold:
                    # Here needs foreground check
                    annotation_list.append(
                        (w_val * self.mask_size_ratio, h_val * self.mask_size_ratio,)
                    )

#         random.shuffle(annotation_list)

        return annotation_list
