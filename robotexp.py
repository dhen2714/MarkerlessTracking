"""
robotexp.py
Functions used to handle arguments, load camera data for robot experiment carried out
by Yidi.

"""
import numpy as np
import cv2

def handle_args(args):
# First argument is the study name, second argument is the feature/descriptor type.
# args should be a list.

    featureOptions = ['sift','surf']
    studyOptions = ['yidi_nostamp','andre_nostamp','yidi_stamp1','yidi_stamp2',
                    'andre_stamp1','andre_stamp2']
			
    if len(args) > 1:
        
        if (str(args[1]).lower() in studyOptions) is True:
            study = args[1].lower()
        else:   
            print("Please specify study:\n\n",studyOptions)
            quit()

    else:

        print("Please specify study:\n\n",studyOptions,
              "\n\nand feature type:\n\n",featureOptions)
        quit()
	
    if len(args) > 2:

        if (str(args[2]).lower()) == 'surf':
            featureType = cv2.xfeatures2d.SURF_create()
        elif (str(args[2]).lower()) == 'sift':
            featureType = cv2.xfeatures2d.SIFT_create()
        else:
            print("\nFeature detector type not recognised, try again.\n")
            quit()
    
    else:

        print("Please specify feature type:\n\n",featureOptions)
        quit()
			
    return study, featureType