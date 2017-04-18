"""
28/03/17
Estimate motion using feature detectors.

python MotionTracking_v2.py study detector estmethod

study: [all,yidi_nostamp,yidi_stamp1,yidi_stamp2,
        andre_nostamp,andre_stamp1,andre_stamp2]
detector: [all,sift,surf,brisk,orb]
estmethod: [all,GN, Horn]

The 'all' option loops over all studies and/or detectors.

For functions used, see camerageometry.py, landmarks.py, and robotexp.py.
"""
import sys
import numpy as np
import cv2
import camerageometry as cg
import landmarks as lm
import robotexp
import time

def motion_tracking(filepath,frames,study,featureType,estMethod,matching_param,
                    P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2):
# This function incorporates the main loop of MotionTracking.py.
# Inputs:
#     filepath: Where the folders containing images for each study are.
#     frames: List of frames to process.
#     study: The study, e.g. yidi_nostamp
#     featureType: The feature/descriptor to use.
#     estMethod: For pose estimation, either GN or Horn.
#     matching_param: Matching parameter. For SIFT and SURF, this parameter is 
#     the value of beta used in the ratio test. For ORB and BRISK, it is the
#     Hamming distance above which matches will be rejected. 0.6 is a good
#     value for SIFT, SURF. 40 & 50 are good values for ORB and BRISK
#     respectively.
#     P1 & P2: Camera matrices.
#     fc1 & fc2: Focal points.
#     pp1 & pp2: Principal points.
#     kk1 & kk2: Radial distortion coefficients.
#     kp1 & kp2: Tangential distortion coefficients.
#     Tr1 & Tr2: 3x3 rectifying transforms.
# Outputs:
#     pList: Nx6 array of poses, where N is the number of frames processed.
#     pList: Nx6 array of poses, where N is the number of frames processed.
#     lms_record: Nx3 array, the first column is a record of the number of 
#     landmarks in the database, the second column is the number of 
#     features/points used in the calculation of the pose estimate. The third
#     column is a series of 0 or -1 depending on whether or not the pose
#     estimation was successful or within reason for that frame.
#     process_time: Processing time.

    n_frames = len(frames)
    pList = np.zeros((n_frames,6))
    lms_record = np.zeros((n_frames,3))
    
    # Initialize matcher, brute force matcher in this case.
    if featureType in ['sift','surf']:
    # For SIFT and SURF, use nearest neighbour matching and L2 NORM.
        bf = cv2.BFMatcher(cv2.NORM_L2)
        desType = np.float32
    else:
    # ORB and BRISK use binary descriptors, use HAMMING norm instead of L2.
    # crossCheck makes sure matches are mutually the best for each set of
    # keypoints.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        desType = np.uint8
        
    if featureType == 'sift':
        fT = cv2.xfeatures2d.SIFT_create()
    elif featureType == 'surf':
        fT = cv2.xfeatures2d.SURF_create()
    elif featureType == 'brisk':
        fT = cv2.BRISK_create()
    elif featureType == 'orb':
        fT = cv2.ORB_create()

    start = time.perf_counter()
    i = 0 # Iteration number. This may not necessarily be the same as frame.
    
    # Main loop.
    for frame in frames:
    
        print("Processing {}, frame number {}...\n".format(study,frame))
        img1 = cv2.imread(filepath+r'/'+study+r'/'+
                          'cam769_pos{}.pgm'.format(frame),0)
        img2 = cv2.imread(filepath+r'/'+study+r'/'+
                          'cam802_pos{}.pgm'.format(frame),0)
           
        (key1, des1) = fT.detectAndCompute(img1,None)
        (key2, des2) = fT.detectAndCompute(img2,None)
        
        # Length of descriptor is dependent on feature type.
        des_len = des1.shape[1]
    
        # Assembling arrays of pixel coords for keypoints in current frame. 
        c1px = np.zeros((len(key1),2),dtype=np.float32)
        c2px = np.zeros((len(key2),2),dtype=np.float32)
        c1px[:,:] = np.array([key1[j].pt for j in range(len(key1))])
        c2px[:,:] = np.array([key2[j].pt for j in range(len(key2))])

        # Correct for distortion.
        c1px = cg.correct_dist(c1px,fc1,pp1,kk1,kp1)
        c2px = cg.correct_dist(c2px,fc2,pp2,kk2,kp2)

        # Intra-frame matching - create a list of 'DMatch' objects, which can 
        # be queried to obtain matched keypoint indices and their 
        # spatial positions.
        if featureType in ['sift','surf']:
            match = bf.knnMatch(des1,des2,k=2)
            matchProper = []
            
            for m, n in match:
                if m.distance < matching_param*n.distance:
                    matchProper.append(m)
                        
        # Remove duplicate (unreliable) matches.
            matchProper = lm.remove_duplicates(np.array(matchProper))
        else:
        # For ORB and BRISK.
            match = bf.match(des1,des2)
            matchProper = lm.binary_thresh(match,matching_param)
              
        # Obtain indices of intra-frame matches.
        in1 = np.array([matchProper[j].queryIdx 
                        for j in range(len(matchProper))],dtype='int')
        in2 = np.array([matchProper[j].trainIdx
                        for j in range(len(matchProper))],dtype='int')
                    
        # Remove indices that don't satisfy epipolar constraint.
        inepi = cg.epipolar_constraint(c1px[in1],c2px[in2],Tr1,Tr2)
        in1 = in1[inepi]
        in2 = in2[inepi]
    
        # Triangulate verified points.
        X = cv2.triangulatePoints(P1,P2,c1px[in1].T,c2px[in2].T)
        X = np.apply_along_axis(lambda v: v/v[-1],0,X)
        X = X[:3,:].T
    
        # framePos is an Nx4 augmented array of 3D triangulated keypoint
        # positions. frameDes are the corresponding descriptors of these
        # intra-frame matches.
        framePos = np.ones((len(inepi),4),dtype=np.float32)
        frameDes = np.ones((len(inepi),des_len),dtype=desType)
        framePos[:,:3] = X
        if featureType in ['sift','surf']:
            frameDes = (des1[in1] + des2[in2])/2
            des1[in1] = (des1[in1] + des2[in2])/2
            des2[in2] = (des1[in1] + des2[in2])/2
        else:
            frameDes = des1[in1]
        
        # Database matching. If it is the first frame, the database is a copy 
        # of frameDes.
        if i == 0:
            dbPos = np.copy(framePos)
            dbDes = np.copy(frameDes)
            dbPos_matched = dbPos
            dbDes_matched = dbDes
            pEst = [0,0,0,0,0,0]

        elif estMethod == 'Horn':
            frameIdx, dbIdx = lm.dbmatch3D(frameDes,dbDes,featureType,matching_param)
            # Horn's method needs at least 3 points in each frame.
            if (len(frameIdx) >= 3 and len(dbIdx) >= 3):
                framePos_matched = framePos[frameIdx]
                frameDes_matched = frameDes[frameIdx]
                dbPos_matched = dbPos[dbIdx]
                dbDes_matched = dbDes[dbIdx]
            else: 
                print("Not enough matches with database, returning previous pose.\n")
                pList[i,:] = pEst
                lms_record[i,0] = dbPos.shape[0]
                lms_record[i,1] = 0
                lms_record[i,2] = -1
                i += 1
                continue
        
        elif estMethod == 'GN':
            indb1, dbm1, indb2, dbm2 = lm.dbmatch(des1,des2,
                                                  dbDes,featureType,matching_param)
	
        # Estimate pose. Points in frameDes that are not matched with landmarks in
        # the database are added to database.
        if i != 0:
        
            if estMethod == 'Horn':
                H = cg.hornmm(framePos_matched,dbPos_matched)
                pEst = cg.mat2vec(H)
                # Outlier removal.
                sqerr = np.sqrt(np.sum(np.square((framePos_matched.T 
                                - np.dot(H,dbPos_matched.T))),0))

                outliers = lm.detect_outliers(sqerr)
                framePos_matched = np.delete(framePos_matched,outliers,axis=0)
                H = cg.hornmm(framePos_matched,
                            np.delete(dbPos_matched,outliers,axis=0))
                dbPos = np.delete(dbPos,dbIdx[outliers],axis=0)
                dbDes = np.delete(dbDes,dbIdx[outliers],axis=0)
            
                # Record number of points used to estimate pose.
                lms_record[i,1] = framePos_matched.shape[0]
                pflag = 0
            
                # Add new entries to database:
                framePos_new = np.delete(framePos,[frameIdx],axis=0)
                frameDes_new = np.delete(frameDes,[frameIdx],axis=0)
                framePos_new = cg.mdot(np.linalg.inv(H),framePos_new.T).T
                dbPos = np.append(dbPos,framePos_new,axis=0)
                dbDes = np.append(dbDes,frameDes_new,axis=0)
        
            if estMethod == 'GN':
                # In the case of no matches with database, indb1 or indb2 will be
                # empty, which would cause error if called as indices.
                if (len(indb1) and len(indb2)):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,c1px[indb1,:],
                                                c2px[indb2,:],dbPos[dbm1,:],
                                                dbPos[dbm2,:],pEst)
                elif len(indb1):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,c1px[indb1,:],
                                                np.array([]),dbPos[dbm1,:],
                                                np.array([]),pEst)
                elif len(indb2):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,np.array([]),
                                                c2px[indb2,:],np.array([]),
                                                dbPos[dbm2,:],pEst)
                else:
                    pflag = -1
                    n_est = 0
                    print("No matches with database, returning previous pose.\n")
                
                H = cg.vec2mat(pEst)
            
                # Record number of points used to estimate pose.
                lms_record[i,1] = n_est
                lms_record[i,2] = pflag
            
                lmInd = []
                # Add new entries to database:
                if pflag == 0:
                    for j in range(len(in1)):
                        if (in1[j] not in indb1) and (in2[j] not in indb2):
                            lmInd.append(j)
                    framePos_new = framePos[lmInd,:]
                    frameDes_new = frameDes[lmInd,:]
                    framePos_new = cg.mdot(np.linalg.inv(H),framePos_new.T).T
                    dbPos = np.append(dbPos,framePos_new,axis=0)
                    dbDes = np.append(dbDes,frameDes_new,axis=0)

        print("Pose estimate for frame {} of {} is:\n {} \n".format(frame,study,pEst))
        pList[i,:] = pEst

        nlm = dbPos.shape[0]
        lms_record[i,0] = nlm
    
        print("{} landmarks in database.\n".format(nlm))

        # Update iteration number
        i += 1

    process_time = time.perf_counter() - start
    
    print("Time taken to process study {}: {}\n\n".format(study,process_time))
    
    return pList, lms_record, process_time

    
if __name__ == '__main__':
    import datetime
    today = datetime.date.today()
    imgPath = (r'C:/Users/dhen2714/Documents/PHD/Experiments/YidiRobotExp/'+
               r'robot_experiment/images')

    # Load camera matrices.
    P = np.fromfile(r'C:/Users/dhen2714/Documents/PHD/Experiments/'+
                    r'YidiRobotExp/robot_experiment/Pmatrices_robot_frame.dat',
                    dtype=float,count=-1)
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
    
    studies, featureTypes, estMethods = robotexp.handle_args_v2(sys.argv)
    
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
    
    output_path = r'C:/Users/dhen2714/Documents/PHD/Experiments/YidiRobotExp/Results/Test/'
    
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
            if featureType == 'brisk':
                matching_param = 50
            elif featureType == 'orb':
                matching_param = 40
            else:
                matching_param = 0.6
        
            for estMethod in estMethods:
                pList, lms_record, process_time = motion_tracking(imgPath,frames,study,featureType,estMethod,matching_param,P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2)
                data_array = np.hstack((pList,lms_record))
                file_path = output_path + study + r'/'
                filename = file_path + 'data_{0}_{1}_{2}.txt'.format(study,featureType,estMethod)
                Header = 'Study: {}\nFeatures: {}\nPoses processed: {}'.format(study,featureType,frames)
                Footer = 'Date of data output: {}\nTime taken: {}'.format(today,process_time)
                np.savetxt(filename,data_array,header=Header,footer=Footer)
                
        
    tot_time = time.perf_counter() - tot_start
    
    print("The total processing time is: {}".format(tot_time))