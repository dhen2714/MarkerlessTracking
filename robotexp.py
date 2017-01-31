"""
robotexp.py
Functions used to handle arguments, load camera data for robot experiment 
carried out by Yidi.

"""
import numpy as np
import cv2

def handle_args(args):
# First argument is the study name, second argument is the feature/descriptor
# type, third argument is pose estimation method.
# args should be a list.

    featureOptions = ['sift','surf']
    studyOptions = ['yidi_nostamp','andre_nostamp','yidi_stamp1',
                    'yidi_stamp2','andre_stamp1','andre_stamp2']
    posEstOptions = ['horn','gn']
    
    if len(args) != 4:
        print("Incorrect number of arguments, specify study, feature type" + 
              " and pose estimation method. E.g.:\n\n" + 
              "python MotionTracking.py yidi_nostamp sift GN\n\n")
        quit()
			
    if (str(args[1]).lower() in studyOptions) is True:
        study = args[1].lower()
    else:   
        print("Study not recognised, specify one of:\n\n",studyOptions)
        quit()

    if (str(args[2]).lower() in featureOptions) is True:
        if (str(args[2]).lower()) == 'surf':
            featureType = cv2.xfeatures2d.SURF_create(extended=True)
        elif (str(args[2]).lower()) == 'sift':
            featureType = cv2.xfeatures2d.SIFT_create()
    else:
        print("Feature detector type not recognised, specify one of:\n\n",
              featureOptions)
        quit()
        
    if (str(args[3]).lower() in posEstOptions) is True:
        estMethod = args[3].lower()
    else:
        print("Pose estimation method not recognised, specify either Horn" +
              " or GN.")
        quit()
		
    return study, featureType, estMethod