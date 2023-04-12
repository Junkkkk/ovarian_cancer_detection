import cv2
import numpy as np

from sklearn.model_selection import train_test_split


def file_split(list_files, labels, train_frac = 0.85, random_state=0):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2

    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
                                        list_files, labels, train_size=train_frac, random_state=random_state)

    X_val, X_test, Y_val, Y_test   = train_test_split(
                                        X_tmp, Y_tmp, train_size=0.5, random_state=random_state)

    return X_train, X_val, X_test, # Y_train, Y_val, Y_test


def calc_white_background_old(image):
    '''
        Return: Percent of white pixel (float, xx.xxx)
    '''
    image_threshold = [240, 240, 240]
    
    #path_img = '{}/{}'.format(path_patch, image_name)
    #image = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
    total_pixel_count = image.shape[0] * image.shape[1]
    white_count = 0
    
    for y_pixel in range(0, image.shape[1]):
        for x_pixel in range(0, image.shape[0]):
            if ( image[x_pixel, y_pixel] >= np.array(image_threshold) ).all():
                white_count = white_count + 1
                
    percent = ( float(white_count) / float(total_pixel_count) ) * 100

    return round(percent, 3)

def find_white_background(image):
    '''
        Return: Percent of white pixel (float, xx.xxx)
    '''
    array_threshold = np.full((512,512,3), [240, 240, 240])
    
    total_pixel_count = image.shape[0] * image.shape[1] * image.shape[2]
    
    percent = ( float(np.sum(image >= array_threshold)) / float(total_pixel_count) ) * 100
    #print(image_name, "'s white background ratio: ", percent)

    return round(percent, 3)

def find_white_background_200(image):
    '''
        Return: Percent of white pixel (float, xx.xxx)
    '''
    array_threshold = np.full((512,512,3), [200, 200, 200])
    
    total_pixel_count = image.shape[0] * image.shape[1] * image.shape[2]
    
    percent = ( float(np.sum(image >= array_threshold)) / float(total_pixel_count) ) * 100
    #print(image_name, "'s white background ratio: ", percent)

    return round(percent, 3)
