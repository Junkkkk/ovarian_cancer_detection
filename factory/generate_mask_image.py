import xmltodict
import json
import numpy as np
import cv2
import os
import shutil
import openslide
from PIL import Image
from syntax.slide import Slide
from syntax.transformers.tissue_mask import OtsuTissueMask

import argparse


class XML:
    def __init__(self, xml_path, wsi_path, mask_path, file_name, level=4):
        self.xml_path = xml_path
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.level = level
        self.file_name = file_name
        self.raw_info, self.contour = self.get_xml()
        self.width, self.height, self.wsi = self.get_wsi()
        self.init_path()

    def get_xml(self):
        anno_dict = {}
        contour_dict = {}
        if os.path.isfile(self.xml_path):
            with open(self.xml_path) as fd:
                xml = json.loads(json.dumps(xmltodict.parse(fd.read())))
                try:
                    if type(xml["ASAP_Annotations"]["Annotations"]["Annotation"]) is dict:
                        xml_info = [xml["ASAP_Annotations"]["Annotations"]["Annotation"]]
                    else:
                        xml_info = xml["ASAP_Annotations"]["Annotations"]["Annotation"]
                    for annotation in xml_info:
                        anno_type = annotation["@PartOfGroup"]
                        contour = []
                        coord_list = []
                        for coordinate in annotation["Coordinates"]["Coordinate"]:
    
                            x = float(coordinate["@X"])
                            y = float(coordinate["@Y"])
                            coord_list.append({"X": x, "Y": y})
                            ##contour.append([int(y), int(x)])
                            contour.append([int(x), int(y)])
    
                        if anno_type in anno_dict:
                            anno_dict[anno_type].append(coord_list)
                            contour_dict[anno_type].append(np.array(contour))
                        else:
                            anno_dict[anno_type] = [coord_list]
                            contour_dict[anno_type] = [np.array(contour)]
                except KeyError:
                    if type(xml["Annotations"]["Annotation"]["Regions"]["Region"]) is dict:
                        xml_info = [xml["Annotations"]["Annotation"]["Regions"]["Region"]]
                    else:
                        xml_info = xml["Annotations"]["Annotation"]["Regions"]["Region"]
                    for annotation in xml_info:
                        anno_type = annotation["@InputRegionId"]
                        contour = []
                        coord_list = []
                        for coordinate in annotation["Vertices"]["Vertex"]:
    
                            x = float(coordinate["@X"])
                            y = float(coordinate["@Y"])
                            coord_list.append({"X": x, "Y": y})
                            ##contour.append([int(y), int(x)])
                            contour.append([int(x), int(y)])
    
                        if anno_type in anno_dict:
                            anno_dict[anno_type].append(coord_list)
                            contour_dict[anno_type].append(np.array(contour))
                        else:
                            anno_dict[anno_type] = [coord_list]
                            contour_dict[anno_type] = [np.array(contour)]
        return anno_dict, contour_dict

    def get_wsi(self):
        wsi = openslide.OpenSlide(self.wsi_path)
        width, height = wsi.level_dimensions[self.level]
        return width, height, wsi

    def init_path(self):
        if os.path.isdir(os.path.join(self.mask_path, file_name)):
            shutil.rmtree(os.path.join(self.mask_path, file_name))
            os.mkdir(os.path.join(self.mask_path, file_name))
        else:
            os.mkdir(os.path.join(self.mask_path, file_name))

    def generate_tissue_mask(self):
        syntax_slide = Slide(slide_path=self.wsi_path)
        Otsu = OtsuTissueMask()

        syntax_slide = Otsu.transform(syntax_slide)

        tissue_mask = syntax_slide.tissue_mask

        tissue_img = Image.fromarray(tissue_mask.data.astype(float))

        tissue_img.thumbnail(size=(int(self.width / pow(2, 5)), int(self.height / pow(2, 5))))

        tissue_img = np.asarray(tissue_img) * 255
        # tissue_img = cv2.resize( tissue_img, (int(self.width / pow(2, 5)), int(self.height / pow(2, 5))) )

        cv2.imwrite(os.path.join(self.mask_path, self.file_name, "tissue.png"), tissue_img)

    def generate_xml_mask(self):
        for label in self.contour:
            drawing = np.zeros([self.height, self.width], np.uint8)
            for cnt in self.contour[label]:
                cv2.drawContours(drawing, [cnt], 0, (255), thickness=-1)

            drawing = cv2.resize( drawing, dsize=(int(self.width / pow(2, 5)), int(self.height / pow(2, 5))), interpolation=cv2.INTER_LINEAR )

            label = "Lesion"
            cv2.imwrite(os.path.join(self.mask_path, self.file_name, label + ".png"), drawing)

    def generate_xml_mask_adj(self):        
        anno = cv2.imread(os.path.join(self.mask_path,self.file_name,"Lesion.png"))
        tissue_mask = cv2.imread(os.path.join(self.mask_path,self.file_name,"tissue.png"))
        
        if anno.shape != tissue_mask.shape:
            anno=cv2.resize(anno,dsize=(tissue_mask.shape[1], tissue_mask.shape[0]),interpolation=cv2.INTER_LINEAR)

        tissue_mask = anno * tissue_mask
        tissue_mask = cv2.threshold(tissue_mask,0.1,255,cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(self.mask_path, self.file_name, "Lesion_adj.png"), tissue_mask)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, required=True, help='/home/Dataset/BRCA/ovary/raw_images')
    parser.add_argument('--mask_dir', type=str, required=True, help='/home/Dataset/BRCA/ovary/mask_images')
    parser.add_argument('--annotation', type=str, required=True, help='True or False')
    config = parser.parse_args()

    root = config.wsi_dir
    mask_path = config.mask_dir
    if not os.path.exists(mask_path):
		    os.makedirs(mask_path)

    files = os.listdir(root)
    files = [f.split(".")[0] for f in files]

    files = list(set(files))
    for file_name in files:
        wsi_path = os.path.join(root, file_name + ".svs")
        xml_path = os.path.join(root, file_name + ".xml")
        out_path = os.path.join(mask_path, file_name)
        tissue_path = os.path.join(out_path, "tissue.png")
        if os.path.isfile(wsi_path):
            print("current: ", file_name)
            xml = XML(
                xml_path=xml_path,
                wsi_path=wsi_path,
                mask_path=mask_path,
                file_name=file_name,
                level=0,
            )
            if config.annotation == "True":
                if os.path.isfile(xml_path):
                    xml.generate_xml_mask()
                    xml.generate_tissue_mask()
                    xml.generate_xml_mask_adj()
                else:
                    xml.generate_tissue_mask()  
            elif config.annotation == "False":
                xml.generate_tissue_mask()
        else:
            print(file_name)
