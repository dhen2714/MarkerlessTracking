"""
02/08/17
Estimate motion using feature detectors.

python MotionTracking_v2.py study detector estmethod

study: [all,yidi_nostamp,yidi_stamp1,yidi_stamp2,
        andre_nostamp,andre_stamp1,andre_stamp2]
detector: [all,sift,surf,brisk,orb,akaze]
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

def load_image(cam=1,study='yidi_nostamp',frame=0,
               imgPath=r'C:/Users/dhen2714/Documents/PHD/'+
               r'Experiments/YidiRobotExp/robot_experiment/images/')):
    if cam == 1:
        cam_name = 'cam769_pos{}.pgm'.format(frame)
    elif cam == 2:
        cam_name = 'cam802_pos{}.pgm'.format(frame)
        
    img = cv2.imread(imgPath+study+r'/'+cam_name,0)
    
    return img

def get_keypoints_descriptors(img,detectorType,descriptorType):
    keys = detectorType.detect(img)
    keys, descriptors = descriptorType.compute(img)
    
    return keys, descriptors
    
def get_pixel_coords(keypoints,fc,pp,kk,kp):
    """Get pixel coordinates of keypoints and correct for distortion."""
    px = np.array([keypoints[j].pt for j in range(len(keypoints))])
    px = cg.correct_dist(px,fc,pp,kk,kp)
    
    return px
    
def matched_indices(des1,des2,ratioTest):
    """Perform matching. If ratioTest is True, use nearest neighbour ratio 
    test. If not, perform binary descriptor matching with crossChecking.
    Returns indices of descriptor matches."""
    if ratioTest:
        bf = cv2.BFMatcher()
        match = bf.knnMatch(des1,des2,k=2)
        matches = []
        
        for m, n in match:
            if m.distance < 0.6*n.distance:
                matches.append(m)
                
    else:
        bf = cv2.BFMatcher(cv2.NORM_Hamming, crossCheck=True)
        matches = bf.match(des1,des2)
        
    matches = lm.remove_duplicates(np.array(matches))
    in1 = np.array([matchProper[j].queryIdx 
                    for j in range(len(matchProper))],dtype='int')
    in2 = np.array([matchProper[j].trainIdx
                    for j in range(len(matchProper))],dtype='int')
        
    return in1, in2
    
def triangulate_keypoints(P1,P2,c1px,c2px):
    """Outputs Nx4 array of homogeneous triangulated matched keypoint 
    positions."""
    X = cv2.triangulatePoints(P1,P2,c1px[in1].T,c2px[in2].T)
    X = np.apply_along_axis(lambda v: v/v[-1],0,X)
    
    return X.T
    
class landmark_database:
    """Database which contains 3D landmark positions and descriptors."""
    
    def __init__(self,X=np.array([]),descriptors=np.array([])):
        self.landmarks = X
        self.descriptors = descriptors
        
    def isempty(self):
        if len(self.landmarks):
            flag = False
        else
            flag = True 
        return flag
        
    def update(self,X,descriptors):
        self.landmarks = np.vstack((self.landmarks,X))
        self.descriptors = np.vstack((self.descriptors,descriptors))
        
    def trim(self,indices):
        self.landmarks = np.delete(self.landmarks,indices,axis=0)
        self.descriptors = np.delete(self.descriptors,indices,axis=0)
        
    def num_elements(self):
        return len(self.landmarks)
        

def calc_pose_Horn(X,descriptors,database,prev_pose,ratioTest=True):
    """Estimate pose using Horn's method."""
    # Database matching.
    frameIdx, dbIdx = matched_indices(descriptors,database.descriptors,ratioTest)
    
    # Pose estimation.
    if (len(frameIdx) >= 3 and len(dbIdx) >= 3):
        X_matched = X[frameIdx]
        descriptors_matched = descriptors[frameIdx]
        dbPos_matched = database.landmarks[dbIdx]
        dbDes_matched = database.descriptors[dbIdx]
        H = cg.hornmm(X_matched,dbPos_matched)
        pose_est = cg.mat2vec(H)
        
        #Outlier removal, recalc pose.
        sqerr = np.sqrt(np.sum(np.square((X_matched.T 
                - np.dot(H,dbPos_matched.T))),0))
        outliers = lm.detect_outliers(sqerr)
        X_matched = np.delete(X_matched,outliers,axis=0)
        H = cg.hornmm(X_matched,
                    np.delete(dbPos_matched,outliers,axis=0))
        database.trim(dbIdx[outliers])

        # Add new entries to database.
        X_new = np.delete(X,[frameIdx],axis=0)
        descriptors_new = np.delete(descriptors,[frameIdx],axis=0)
        X_new = cg.mdot(np.linalg.inv(H),X_new.T).T
        database.update(X_new,descriptors_new)
    else:
        print("Not enough matches with database, returning previous pose.\n")
        pose_est = prev_pose 

    return pose_est, database
    
def calc_pose_GN(c1px,c2px,d1,d2,database,prev_pose,X,in1,in2,
                 P1,P2,ratioTest=True):
    """Estimate pose using GN iterations."""
    # Database matching.
    fIdx1, dbIdx1 = matched_indices(d1,database.descriptors,ratioTest)
    fIdx2, dbIdx2 = matched_indices(d2,database.descriptors,ratioTest)
    
    # Pose estimation.
    if (len(fIdx1) and len(lenfIdx2)):
        pose_est, n_est, pflag = lm.GN_estimation(
                                 P1,P2,
                                 c1px[fIdx1,:],
                                 c2px[fIdx2,:],
                                 database.landmarks[dbIdx1,:],
                                 database.landmarks[dbIdx2,:],
                                 prev_pose)
    elif len(indb1):
        pose_est, n_est, pflag = lm.GN_estimation(
                                 P1,P2,
                                 c1px[fIdx1,:],
                                 np.array([]),
                                 database.landmarks[dbIdx1,:],
                                 np.array([]),
                                 prev_pose)
    elif len(indb2):
        pose_est, n_est, pflag = lm.GN_estimation(
                                 P1,P2,
                                 np.array([]),
                                 c2px[fIdx2,:],
                                 np.array([]),
                                 database.landmarks[dbIdx2,:],
                                 prev_pose)
    else:
        pflag = -1
        n_est = 0
        print("No matches with database, returning previous pose.\n")
        
    H = cg.vec2mat(pose_est)
    # Add new entries to database.
    new_lms =[]
    if pflag == 0:
        for j in range(len(in1)):
            if (in1[j] not in fIdx1) and (in2[j] not in fIdx2):
                new_lms.append(j)
                
        X_new = X[new_lms,:]
        descriptors_new = d1[new_lms,:]
        X_new = cg.mdot(np.linalg.inv(H),X_new.T).T
        database.update(X_new,descriptors_new)
        
    return pose_est, database

def process_frame(frame,prev_pose,study,detectorType,descriptorType,estMethod,
                  database,ratioTest=True,
                  P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2):
    """Output pose estimate for the current frame."""
    print("Processing {}, frame number {}...\n".format(study,frame))
    img1 = load_image(1,study,frame)
    img2 = load_image(2,study,frame)

    k1, d1 = get_keypoints_descriptors(img1,detectorType,descriptorType)
    k2, d2 = get_keypoints_descriptors(img2,detectorType,descriptorType)
    
    # Assemble arrays of pixel coordinates for keypoints in current frame.
    c1px = get_pixel_coords(k1,fc1,pp1,kk1,kp1)
    c2px = get_pixel_coords(k2,fc2,pp2,kk2,kp2)
    
    # Intra-frame matching. Creates a list of 'DMatch' objects, which can be 
    # queried to obtain matched keypoint indices and their positions.
    in1, in2 = matched_indices(d1,d2,ratioTest)
    
    # Remove indices that don't satisfy epipolar constraint.
    inepi = cg.epipolar_constraint(c1px[in1],c2px[in2],Tr1,Tr2)
    in1 = in1[inepi]
    in2 = in1[inepi]
    # Triangulate intra-frame matched keypoints.
    X = triangulate_keypoints(P1,P2,c1px[in1],c2px[in2])
    
    if frame == 1 or database.isempty():
        database = landmark_database(X,d1[in1])
        pose_est = prev_pose
    elif estMethod == 'Horn':
        pose_est, database = calc_pose_Horn(X,d1[in1],database,prev_pose,ratioTest)
    elif estMethod == 'GN':
        pose_est, database = calc_pose_GN(c1px,c2px,d1,d2,database,prev_pose,
                                          X,in1,ind2,P1,P2,ratioTest)

    print("Pose estimate for frame {} of {} is:\n {} \n".format(frame,study,pose_est))
    
    print("{} landmarks in database.\n".format(database.num_elements()))

    return pose_est, database

def main(filepath,frames,study,detectorType,descriptorType,estMethod,
                    P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,kp1,kp2,Tr1,Tr2):
# This function incorporates the main loop of MotionTracking.py.
# Inputs:
#     filepath: Where the folders containing images for each study are.
#     frames: List of frames to process.
#     study: The study, e.g. yidi_nostamp
#     featureType: The feature/descriptor to use.
#     estMethod: For pose estimation, either GN or Horn.
#     matching_param: Matching parameter. This parameter is the value of beta 
#     used in the ratio test.
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
#     counts: Nx5 array. First two columns are the number of keypoints detected
#     by each camera, third column is number of intra-frame matches, fourth
#     column is number of points used to estimate pose, fifth is the number of
#     landmarks in database.
#     timings: Nx4 array. First column is time taken for feature detection and
#     extraction for both cameras. Second column is time taken for intra-frame
#     matching. Third column is time taken for database matching. Fourth column
#     is time taken for pose estimation.
#     process_time: Processing time.

    n_frames = len(frames)
    poses = np.zeros((n_frames,6))
    counts = np.zeros((n_frames,5))
    timings = np.zeros((n_frames,4))
    
    i = 0 # Iteration number. This may not necessarily be the same as frame.
    
    # Initiate pose and database.
    pose_est = np.zeros(6)
    database = landmark_database()
    
    start = time.perf_counter()
    # Main loop.
    for frame in frames:
        pose_est, database = process_frame(frame,pose_est,study,detectorType,
                                           descriptorType,estMethod,database,
                                           ratioTest,
                                           P1,P2,fc1,fc2,pp1,pp2,kk1,kk2,
                                           kp1,kp2,Tr1,Tr2)
                                           
        poses[i,:] = pose_est
        i += 1
        
    process_time = time.perf_counter() - start
    
    return
    
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
    
    output_path = r'C:/Users/dhen2714/Documents/PHD/Experiments/YidiRobotExp/Results/20170428_Results/'
    
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