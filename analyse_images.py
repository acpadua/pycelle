import numpy as np

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from pandas import DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import os
from os import path
import glob
import cv2
import time
import math
import json

def convert_to_grayscale(img, file_name, output_gray):
    """
    This function converts the original image in a grayscale image. Then shows and saves the gray scale image.
    :param img: path to image of interest
    :param output_path: path for the output image
    """
    # reads the image img in grayscale
    img_gray = cv2.imread(img,0)
    save_name = file_name + '_gray.jpg'

    cv2.imwrite(os.path.join(output_gray, save_name), img_gray)
    #cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
    #cv2.imshow(file_name, img_gray)
    #cv2.waitKey(2)
    #cv2.destroyAllWindows()
    print("converted to gray")
    return img_gray

def cumsum_histogram_percentage(img, file_name, output_histograms):
    """
    Reads the image 'img', gets the images' histogram and then the cumulative distribution.
    Next, normalises the cumulative distribution to be between 0 and 1. Finally, shows and saves it in a plot.  
    :param img: path to the image of interest 
    :param output_histograms: path where the cumulative distribution histogram will be saved. 
    :return: cumulative distribution histogram, grayscale values, image name 
    """
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_percentage = cdf * 1 / cdf.max()
    plt.plot(cdf_percentage, color = 'b')
    plt.xlim([0,256])
    plt.legend(('cdf'), loc = 'upper left')
    fig1=plt.gcf()
    #plt.show()
    print(img)
    name_fig1 = file_name + '_hist'
    print(name_fig1)
    fig1.savefig(os.path.join(output_histograms, name_fig1))
    print("histogram_done")

    print(cdf_percentage)
    return cdf_percentage, bins

def remove_background(img, file_name, cdf_percentage, bins, output_thresh):
    """
    Finds the value (n) of grayscale where the cumulative distribution of the image's histogram is 75 %.
    Then, applied a threshold (n) to the image, that converts all the values bellow n to 0 (black).
    :param image: path to image of interest
    :param file_name: name of the image of interest
    :param cdf_percentage: array with the values of the cumsum histogram of the image
    :param bins: grayscale values [0, 255]
    :param output_thresh_blur: path where the output image will be saved
    :return: image with threshold of 75 % applied
    """
    #get array with all values > 75 % from cumsum
    third_quartil_cumsum = np.where(cdf_percentage > 0.75)[0]

    # get first value > 75 % from third quartil cumsum, which will be the position where to cut
    position_where_to_cut = third_quartil_cumsum[0]
    print(position_where_to_cut)

    # get gray value where to cut, which is the treshold
    threshold = bins[position_where_to_cut]
    print(threshold)

    ret, img_thresh_75 = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    save_name = file_name + '_thresh75.jpg'

    cv2.imwrite(os.path.join(output_thresh, save_name), img_thresh_75)
    #cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
    #cv2.imshow(save_name, img_thresh_75)
    #cv2.waitKey(2)
    #cv2.destroyAllWindows()
    print("converted to thresh75")
    return img_thresh_75

def remove_white(img, file_name, output_background):
    """
    This function converts the white pixels of the image in black pixels.
    """
    white_px = 220
    black_px = 0

    (row, col) = img.shape
    img_array = np.array(img)

    for r in range(row):
        for c in range(col):
            px = img[r][c]
            if (px > white_px):
                img_array[r][c] = black_px

    print("end for cycle")

    save_name = file_name + '_no_white.jpg'

    cv2.imwrite(os.path.join(output_background, save_name), img_array)
    cv2.namedWindow(save_name, cv2.WINDOW_NORMAL)
    cv2.imshow(save_name, img_array)
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def get_droplets(img, file_name, parameters, output_circles):

    blobs_log1 = blob_log(img, min_sigma= parameters['min_sigma_value'], max_sigma= parameters['max_sigma_value'], num_sigma= parameters['num_sigma_value'], threshold= parameters['threshold_value'])

    blobs_log1[:, 2] = blobs_log1[:, 2] * sqrt(2)
    print("end of blob")

    color = 'lime'

    file_name_parts = file_name.split("_")

    image_description = []

    for part in file_name_parts:
        image_description.append(part)

    x_set = []
    y_set = []
    r_set = []

    droplets = []

    fig1, ax = plt.subplots(1, 1, figsize=(9, 3))

    for blob in blobs_log1:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
        ax.add_patch(c)

        x_set.append(x)
        y_set.append(y)
        r_set.append(r)
    ax.set_axis_off()

    ax.set_title(file_name)
    ax.imshow(img)

    plt.tight_layout()
    fig1 = plt.gcf()
    #plt.show(block=False)
    #time.sleep(0.5)
    #plt.close()


    save_name = file_name + '_min' + str(parameters['min_sigma_value']) + '_max' + str(parameters['max_sigma_value']) + '_num' + str(parameters['num_sigma_value']) + '_thresh' + str(parameters['threshold_value']) + ".svg"
    fig1.savefig(os.path.join(output_circles, save_name), format='svg', dpi=1200)
    return x_set, y_set, r_set

def file_droplets(file_name, output_path, parameters, x_set, y_set, r_set):

    Droplets = {'min_sigma': parameters['min_sigma_value'], 'max_sigma': parameters['max_sigma_value'], 'num_sigma': parameters['num_sigma_value'], 'threshold': parameters['threshold_value'], 'x': x_set, 'y': y_set, 'r': r_set}

    df = DataFrame(Droplets, columns=['min_sigma', 'max_sigma', 'num_sigma', 'threshold', 'x', 'y', 'r'])

    export_csv = df.to_csv(output_path + '/files/' + file_name + '.csv', index=None, header=True)  # Don't forget to add '.csv' at the end of the path

    print (df)

def analyse_droplets(files_path, output_files_analysis):

    voc =[]
    time = []
    label = []

    min_sigma = []
    max_sigma = []
    num_sigma = []
    threshold = []

    droplets_nr = []
    droplets_mean_radius = []
    optical_area = []

    droplets_nr_exp = []
    droplets_mean_radius_exp = []
    optical_area_exp = []

    for file in glob.glob("{}/*csv".format(files_path)):
        df = pd.read_csv(file)
        name_file = os.path.basename(file)
        name_info = name_file.split('_')

        voc.append(name_info[0])
        time.append(name_info[1])
        label.append(name_info[2])

        r_set = ((df['r'] * 5000) / 1850)

        min_sigma.append(df['min_sigma'][0])
        max_sigma.append(df['max_sigma'][0])
        num_sigma.append(df['num_sigma'][0])
        threshold.append(df['threshold'][0])

        count_lines = df.shape[0]
        count_droplets = count_lines +1
        print(str(count_droplets))
        droplets_nr.append(count_droplets)
        droplets_mean_radius.append(r_set.mean())
        optical_area_sensor = 0

        for i in range(count_lines -1):
            optical_area_sensor += (np.pi * (df['r'][i] ** 2))
        optical_area_sensor_relative = (optical_area_sensor / (np.pi * (2500 ** 2)))
        optical_area.append(optical_area_sensor_relative)
        print(str(optical_area_sensor))



    Analysis = {'voc': voc, 'time': time, 'label': label, 'min_sigma': min_sigma, 'max_sigma': max_sigma, 'num_sigma': num_sigma,
            'threshold': threshold, 'droplets_nr': droplets_nr, 'droplets_mean_radius': droplets_mean_radius, 'optical_area': optical_area}

    df = DataFrame(Analysis, columns=['voc', 'time', 'label', 'min_sigma', 'max_sigma', 'num_sigma', 'threshold', 'droplets_nr', 'droplets_mean_radius', 'optical_area'])

    save_name = 'droplets_analysis.csv'
    analysis_file = df.to_csv(os.path.join(output_files_analysis, save_name), index=None, header=True)
    return (analysis_file)

def sort_file(file, output_path):

    df = pd.read_csv(file)

    result = df.sort_values(['voc', 'time', 'label'], ascending=[1, 1, 1])
    result.to_csv(os.path.join(output_path, 'droplets_analysis_sorted.csv'))

    print(result)

def add_columns_validation(file_data, file_validation, output_path):
    df_a = pd.read_csv(file_data)
    df_e = pd.read_csv(file_validation)

    df_a['droplets_nr_exp'] = df_e['droplets_nr_exp']
    df_a['mean_diameter_exp'] = df_e['mean_diameter_exp']

    save_name = 'droplets_analysis_complete.csv'
    complete_file = df_a.to_csv(os.path.join(output_path, save_name))
    return complete_file

def compare_results(complete_file, output_path):

    df_c = pd.read_csv(complete_file)
    compare_nr = []
    compare_radius = []
    error = []

    for i, row in df_c.iterrows():
        sub_nr_i = (row['droplets_nr'] - row['droplets_nr_exp']) / row['droplets_nr_exp']
        compare_nr.append(sub_nr_i)
        print(sub_nr_i)

        mean_radius_i_a = (row['droplets_mean_radius'] * 5000) / 1400
        mean_radius_i_e = (row['mean_diameter_exp'] / 2)
        sub_radius_i = ((mean_radius_i_a - mean_radius_i_e) / mean_radius_i_e)
        print(sub_radius_i)
        compare_radius.append(sub_radius_i)

        error_i = math.sqrt((sub_radius_i ** 2) + (sub_radius_i ** 2))
        print(error_i)
        error.append(error_i)

    df_c['comparison_nr'] = compare_nr
    df_c['compare_raius'] = compare_radius
    df_c['error'] = error

    save_name = 'droplets_analysis_comparison.csv'
    complete_file = df_c.to_csv(os.path.join(output_path, save_name))
    return complete_file

def read_parameters(file_name):
    if(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

def main():
    #from pudb.remote import set_trace; set_trace(term_size=(160, 40), host='0.0.0.0', port=6900)
    input_dir = os.path.abspath('./data')
    output_dir = os.path.abspath('./data/output')
    if(path.exists(output_dir)):
        print("Output folder already exists. Please, remove output folders.")
        return None
        #last_char = output_dir[:-1]
        #if(last_char.isdigit()):
        #    nr = int(last_char) + 1
        #    os.mkdir(os.path.abspath('/data/output' + '_' + str(nr)))
        #    output_path = os.path.abspath('/data/output' +'_' + str(nr))
        #else:
        #    nr = 2
        #    os.mkdir(os.path.abspath('/data/output' + '_' + str(nr)))
        #    output_path = os.path.abspath('/data/output' +'_' + str(nr))
    else:
        os.mkdir(output_dir)
        output_path = output_dir 


    output_gray = os.path.join(output_path, 'gray/')
    os.mkdir(output_gray)

    output_thresh = os.path.join(output_path, 'thresh/')
    os.mkdir(output_thresh)
    output_files = os.path.join(output_path, 'files/')
    os.mkdir(output_files)
    output_files_analysis = os.path.join(output_path, 'files_analysis/')
    os.mkdir(output_files_analysis)

    output_histograms = os.path.join(output_path, 'histogram/')
    os.mkdir(output_histograms)

    output_circles = os.path.join(output_path, 'circles/')
    os.mkdir(output_circles)

    for img in glob.glob("{}/*jpg".format(input_dir)):
        file_name = os.path.basename(img)[:-4]
        parameters = read_parameters(img[:-4] + '.json')

        print(file_name)
        img_gray = convert_to_grayscale(img, file_name, output_gray)
        cdf_percentage, bins = cumsum_histogram_percentage(img_gray, file_name, output_histograms) #confirm if histogram is well done
        img_thresh75 = remove_background(img_gray, file_name, cdf_percentage, bins, output_thresh)

        x_set, y_set, r_set = get_droplets(img_thresh75, file_name, parameters, output_circles)
        file_droplets(file_name, output_path, parameters, x_set, y_set, r_set)
        analyse_droplets(output_files, output_files_analysis)
    #analysis_file = 'droplets_analysis.csv'
    #sort_file(os.path.join(output_files_analysis, analysis_file), output_files_analysis)
    #analysis_file_sorted = 'droplets_analysis_sorted.csv'
    #validation_file = 'vocs_expected_sorted.csv'

    #complete_file = add_columns_validation(os.path.join(output_files_analysis, analysis_file_sorted), os.path.join(output_files_analysis, validation_file), output_files_analysis)
    #complete_file = os.path.join(output_files_analysis, 'droplets_analysis_complete.csv')
    #validation_file = os.path.join(output_files_analysis, validation_file)
    #compare_results(complete_file, output_files_analysis)



if __name__ == "__main__":
    main()





#for i in range(nr_images):
