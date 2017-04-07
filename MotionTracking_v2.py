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

def motion_tracking(filepath,frames,study,featureType,estMethod,P1,P2,fc1,fc2,
                    pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2):
# This function incorporates the main loop of MotionTracking.py.
# Inputs:
#     filepath: Where the folders containing images for each study are.
#     frames: List of frames to process.
#     study: The study, e.g. yidi_nostamp
#     featureType: The feature/descriptor to use.
#     estMethod: For pose estimation, either GN or Horn.
#     P1 & P2: Camera matrices.
#     fc1 & fc2: Focal points.
#     pp1 & pp2: Principal points.
#     kk1 & kk2: Radial distortion coefficients.
#     kp1 & kp2: Tangential distortion coefficients.
#     Tr1 & Tr2: 3x3 rectifying transforms.
# Outputs:
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
    bf = cv2.BFMatcher()
    # Beta value of less than 0.8 for orb leads to bad results. Other feature
    # types are not so sensitive.
    if featureType == 'orb':
        beta = 0.8
    else:
        beta = 0.6
    beta1 = beta # NN matching parameter for intra-frame matching.
    beta2 = beta # For database matching.
    
    start = time.perf_counter()
    i = 0 # Iteration number. This may not necessarily be the same as frame.
    
    # Main loop.
    for frame in frames:
    
        print("Processing {}, frame number {}...\n".format(study,frame))
        img1 = cv2.imread(filepath+r'/'+study+r'/'+
                          'cam769_pos{}.pgm'.format(frame),0)
        img2 = cv2.imread(filepath+r'/'+study+r'/'+
                          'cam802_pos{}.pgm'.format(frame),0)
        
        if featureType == 'sift':
            fT = cv2.xfeatures2d.SIFT_create()
        elif featureType == 'surf':
            fT = cv2.xfeatures2d.SURF_create()
        elif featureType == 'brisk':
            fT = cv2.BRISK_create()
        elif featureType == 'orb':
            fT = cv2.ORB_create()
    
        (key1, des1) = fT.detectAndCompute(img1,None)
        (key2, des2) = fT.detectAndCompute(img2,None)
        
        # Length of descriptor is dependent on feature type.
        des_len = des1.shape[1]
    
        # Assembling descriptors of form [u,v,[descriptor]] for the current frame, 
        # where (u,v) is the pixel coordinates of the keypoint.
        c1des = np.zeros((len(key1),(des_len+2)),dtype=np.float32)
        c2des = np.zeros((len(key2),(des_len+2)),dtype=np.float32)
        c1des[:,:2] = np.array([key1[j].pt for j in range(len(key1))])
        c1des[:,2:] = des1
        c2des[:,:2] = np.array([key2[j].pt for j in range(len(key2))])
        c2des[:,2:] = des2

        # Correct for distortion.
        c1des[:,:2] = cg.correct_dist(c1des[:,:2],fc1,pp1,kk1,kp1)
        c2des[:,:2] = cg.correct_dist(c2des[:,:2],fc2,pp2,kk2,kp2)

        # Intra-frame matching - create a list of 'DMatch' objects, which can 
        # be queried to obtain matched keypoint indices and their 
        # spatial positions.
        match = bf.knnMatch(des1,des2,k=2)
        matchProper = []
    
        for m, n in match:
        # Nearest neighbour matching.
            if m.distance < beta1*n.distance:
                matchProper.append(m)
    
        # Remove duplicate matches (keypoints that match with more than one
        # keypoint in the other view).
        matchProper = lm.remove_duplicates(np.array(matchProper))
    
        # Obtain indices of intra-frame matches.
        in1 = np.array([matchProper[j].queryIdx 
                        for j in range(len(matchProper))],dtype='int')
        in2 = np.array([matchProper[j].trainIdx
                        for j in range(len(matchProper))],dtype='int')
                    
        # Remove indices that don't satisfy epipolar constraint.
        inepi = cg.epipolar_constraint(c1des[in1,:2],c2des[in2,:2],Tr1,Tr2)
        in1 = in1[inepi]
        in2 = in2[inepi]
    
        # Triangulate verified points.
        X = cv2.triangulatePoints(P1,P2,c1des[in1,:2].T,c2des[in2,:2].T)
        X = np.apply_along_axis(lambda v: v/v[-1],0,X)
        X = X[:3,:].T
    
        # Create an array of descriptors of form [x,y,z,1,[descriptor]]
        # of points triangulated in current frame.
        frameDes = np.ones((len(inepi),(4+des_len)),dtype=np.float32)
        for j in range(len(inepi)):
            frameDes[j,:3] = X[j,:3]
            frameDes[j,4:] = (des1[in1[j]] + des2[in2[j]])/2
                          
        c1des[in1,2:] = frameDes[:,4:]
        c2des[in2,2:] = frameDes[:,4:]
    
        # Database matching. If it is the first frame, the database is a copy 
        # of frameDes.
        if i == 0:
            db = np.copy(frameDes)
            dbMatched = db
            pEst = [0,0,0,0,0,0]

        elif estMethod == 'Horn':
            frameIdx, dbIdx = lm.dbmatch3D(frameDes,db,beta2)
            # Horn's method needs at least 3 points in each frame.
            if (len(frameIdx) >= 3 and len(dbIdx) >= 3):
                frameMatched = frameDes[frameIdx]
                dbMatched = db[dbIdx]
            else: 
                print("Not enough matches with database, returning previous pose.\n")
                pList[i,:] = pEst
                lms_record[i,0] = db.shape[0]
                lms_record[i,1] = 0
                lms_record[i,2] = -1
                i += 1
                continue
        
        elif estMethod == 'GN':
            indb1, dbm1, indb2, dbm2 = lm.dbmatch(c1des[:,2:],c2des[:,2:],
                                                  db[:,4:],beta2)
	
        # Estimate pose. Points in frameDes that are not matched with landmarks in
        # the database are added to database.
        if i != 0:
        
            if estMethod == 'Horn':
                H = cg.hornmm(frameMatched[:,:4],dbMatched[:,:4])
                pEst = cg.mat2vec(H)
                # Outlier removal.
                sqerr = np.sqrt(np.sum(np.square((frameMatched[:,:4].T 
                                - np.dot(H,dbMatched[:,:4].T))),0))

                outliers = lm.detect_outliers(sqerr)
                frameMatched = np.delete(frameMatched,outliers,axis=0)
                H = cg.hornmm(frameMatched[:,:4],
                            np.delete(dbMatched[:,:4],outliers,axis=0))
                db = np.delete(db,dbIdx[outliers],axis=0)
            
                # Record number of points used to estimate pose.
                lms_record[i,1] = frameMatched.shape[0]
                pflag = 0
            
                # Add new entries to database:
                frameNew = np.delete(frameDes,[frameIdx],axis=0)
                frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
                db = np.append(db,frameNew,axis=0)
        
            if estMethod == 'GN':
                # In the case of no matches with database, indb1 or indb2 will be
                # empty, which would cause error if called as indices.
                if (len(indb1) and len(indb2)):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,c1des[indb1,:2],
                                                c2des[indb2,:2],db[dbm1,:4],
                                                db[dbm2,:4],pEst)
                elif len(indb1):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,c1des[indb1,:2],
                                                np.array([]),db[dbm1,:4],
                                                np.array([]),pEst)
                elif len(indb2):
                    pEst, n_est, pflag = lm.GN_estimation(P1,P2,np.array([]),
                                                c2des[indb2,:2],np.array([]),
                                                db[dbm2,:4],pEst)
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
                    frameNew = frameDes[lmInd,:]
                    frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
                    db = np.append(db,frameNew,axis=0)

        print("Pose estimate for frame {} of {} is:\n {} \n".format(frame,study,pEst))
        pList[i,:] = pEst

        nlm = db.shape[0]
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
                    r'YidiRobotExp/robot_experiment/Pmatrices.dat',
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
    
    output_path = r'C:/Users/dhen2714/Documents/PHD/Experiments/YidiRobotExp/20170403_Results/Results_no_cc/'
    
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
        
            for estMethod in estMethods:
                pList, lms_record, process_time = motion_tracking(imgPath,frames,study,featureType,estMethod,P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2)
                data_array = np.hstack((pList,lms_record))
                file_path = output_path + study + r'/'
                filename = file_path + 'data_{0}_{1}_{2}.txt'.format(study,featureType,estMethod)
                Header = 'Study: {}\nFeatures: {}\nPoses processed: {}'.format(study,featureType,frames)
                Footer = 'Date of data output: {}\nTime taken: {}'.format(today,process_time)
                np.savetxt(filename,data_array,header=Header,footer=Footer)
                
        
    tot_time = time.perf_counter() - tot_start
    
    print("The total processing time is: {}".format(tot_time))