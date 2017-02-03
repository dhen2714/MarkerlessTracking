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

    featureOptions = ['sift','surf','orb']
    studyOptions = ['yidi_nostamp','andre_nostamp','yidi_stamp1',
                    'yidi_stamp2','andre_stamp1','andre_stamp2']
    posEstOptions = ['1','2']
    
    if len(args) != 4:
        print("Incorrect number of arguments, specify study, feature type" + 
              " and pose estimation method. E.g.:\n\n" + 
              "python MotionTracking.py yidi_nostamp sift 1\n\n" + 
              "(1 for Horn's method, 2 for Gauss-Newton)")
        quit()
			
    if (str(args[1]).lower() in studyOptions) is True:
        study = args[1].lower()
    else:   
        print("Study not recognised, specify one of:\n\n",studyOptions)
        quit()

    if (str(args[2]).lower() in featureOptions) is True:
        if (str(args[2]).lower()) == 'surf':
            # 'extended=True' means descriptors have length 128, instead of 64
            featureType = cv2.xfeatures2d.SURF_create(extended=True)
        elif (str(args[2]).lower()) == 'sift':
            featureType = cv2.xfeatures2d.SIFT_create()
        elif (str(args[2]).lower()) == 'orb':
            featureType = cv2.ORB_create()
    else:
        print("Feature detector type not recognised, specify one of:\n\n",
              featureOptions)
        quit()
        
    if args[3] in posEstOptions:
        estMethod = int(args[3])
    else:
        print("Pose estimation method not recognised, specify either:\n\n" +
              "1 for Horn's method.\n" + 
              "2 for Gauss-Newton.")
        quit()
		
    return study, featureType, estMethod