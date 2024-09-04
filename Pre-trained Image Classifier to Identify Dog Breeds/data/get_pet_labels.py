#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: DuyLK16
# DATE CREATED:  2024-09-02                                
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir
def get_label(filename):
    """
    Given a filename, returns the appropriate label.
    (ex. filename = 'Boston_terrier_02259.jpg' -> Pet label = 'boston terrier')
    
    Parameters:
        filename - The name of the image file
    
    Returns:
        label - A string with the label of the filename
    """
    return " ".join([word for word in filename.lower().split("_") if word.isalpha()])

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based on the filenames 
    of the image files. The pet image labels are used to check the accuracy 
    of the labels returned by the classifier function, as the filenames 
    contain the true identity of the pet in the image.
    
    Parameters:
        image_dir - The (full) path to the folder of images to be
                    classified by the classifier function (string)
    
    Returns:
        results_dic - Dictionary with 'key' as image filename and 'value' as a 
        list. The list contains the following item:
            index 0 = pet image label (string)
    """
    filename_list = listdir(image_dir)
    results_dic = {name: [get_label(name)] for name in filename_list}
    return results_dic

