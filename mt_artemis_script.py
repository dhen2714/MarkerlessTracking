"""
28/04/17
MotionTracking_v2.py script for use on Artemis.
"""
import sys
sys.path.insert(1,r'/project/RDS-FEI-MRI_MT-RW/David/Software/MarkerlessTracking')
import numpy as np
import cv2
import camerageometry as cg
import landmarks as lm
import robotexp
import time
import datetime

if __name__ == '__main__':
    import datetime
    today = datetime.date.today()
    imgPath = r'/project/RDS-FEI-MRI_MT-RW/David/YidiRobotExp/robot_experiment/images'

    # Load camera matrices.
    P = np.fromfile(r'/project/RDS-FEI-MRI_MT-RW/David/YidiRobotExp/robot_experiment/Pmatrices.dat',dtype=float,count=-1)
    P1 = P[:12].reshape(3,4)
    P2 = P[12:].reshape(3,4)

    fc1 = np.array([1680.18107, 1688.48719]) # Focal points
    fc2 = np.array([1650.43476, 1658.81782])

    pp1 = np.array([305.48666, 263.88465]) # Principal points
    pp2 = np.array([319.43151, 254.58401])

    kk1 = np.array([-0.38144, 0.61668]) # Radial distortion
    kk2 = np.array([-0.38342, 0.58461])

    kp1 = np.array([0.00329, -0.00263]) # Tangential distortion
    kp2 = np.array([0.00115, -0.00274])
    
    studies = ['yidi_nostamp','yidi_stamp1','yidi_stamp2','andre_nostamp','andre_stamp1','andre_stamp2']
    featureTypes = ['sift','surf','orb','brisk','akaze']
    estMethods = ['Horn','GN']
    
    # Rectify for outlier removal with epipolar constraint.
    Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2)
    DD1,DD2 = cg.generate_dds(Tr1,Tr2)
    Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2,DD1,DD2)
    
    # Valid frame indices. Some images in certain studies have been mislabelled,
    # so only certain frames should be processed.
    frames_yns = np.arange(30)
    frames_ys1 = np.array([0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    frames_ys2 = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29])
    frames_ans = np.array([0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    #frames_as1 = np.array([0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29])
    # 3/4/17 - Try this array of indices from as1. Instead of using pos2, use pos3.
    frames_as1 = np.array([0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29])
    frames_as2 = np.arange(30)
    valid_frames = [frames_yns,frames_ys1,frames_ys2,frames_ans,frames_as1,frames_as2]
    
    output_path = r'/project/RDS-FEI-MRI_MT-RW/David/YidiRobotExp/Results/20170428_Results/'
    
    tot_start = time.perf_counter()
    for study in studies:
    
        if study == 'yidi_nostamp':
            frames = frames_yns
        elif study == 'yidi_stamp1':
            frames = frames_ys1
        elif study == 'yidi_stamp2':
            frames = frames_ys2
        elif study == 'andre_nostamp':
            frames = frames_ans
        elif study == 'andre_stamp1':
            frames = frames_as1
        elif study == 'andre_stamp2':
            frames = frames_as2
        
        for featureType in featureTypes:
        
            matching_param = 0.6
        
            for estMethod in estMethods:
                pList, lms_record, timings, process_time = motion_tracking(imgPath,frames,study,featureType,estMethod,matching_param,P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2)
                data_array = np.hstack((pList,lms_record,timings))
                file_path = output_path + study + r'/'
                filename = file_path + 'data_{0}_{1}_{2}.txt'.format(study,featureType,estMethod)
                Header = 'Study: {}\nFeatures: {}\nPoses processed: {}'.format(study,featureType,frames)
                Footer = 'Date of data output: {}\nTime taken: {}'.format(today,process_time)
                np.savetxt(filename,data_array,header=Header,footer=Footer)
                
        
    tot_time = time.perf_counter() - tot_start
    
    print("The total processing time is: {}".format(tot_time))