"""
Estimates pose using feature detectors.

Specify study, feature detector, and pose estimation method.
E.g.:
python MotionTracking.py yidi_nostamp sift 1

Last argument specifies the pose estimation method, and is either 1 for Horn's 
method, or 2 for Gauss-Newton.

For functions used, see robotexp.py, camerageometry.py and landmarks.py

"""
import sys
import numpy as np
import cv2
import camerageometry as cg
import landmarks as lm
import robotexp
import time

imgPath = ("C:\\Users\\dhen2714\\Documents\\PHD\\Experiments" + 
           "\\YidiRobotExp\\robot_experiment\\images\\")

# Load camera matrices.
P = np.fromfile("C:\\Users\\dhen2714\\Documents\\PHD\\Experiments\\"+
                "YidiRobotExp\\robot_experiment\\Pmatrices_robot_frame.dat",
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

poseNumber = 30 # Number of frames to process.
poseList = np.zeros((poseNumber,6))
lmList = np.zeros(poseNumber) # Record of database landmarks from frame to frame.
estList = np.zeros(poseNumber) # How many points were used in estimating pose. 

# Name of study, featureType from user input. E.g., 'yidi_nostamp' 'sift'
study, featureType, beta, estMethod = robotexp.handle_args(sys.argv)
print("\nChosen study is: {}\n\nChosen feature type is: {}\n\n"
      .format(study,sys.argv[2].lower()))
input("Press ENTER to continue.\n\n")

# Initialize matcher, brute force matcher in this case.
bf = cv2.BFMatcher()
beta1 = beta # NN matching parameter for intra-frame matching.
beta2 = beta # For database matching.

# Rectify for outlier removal with epipolar constraint.
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2)
DD1,DD2 = cg.generate_dds(Tr1,Tr2)
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2,DD1,DD2)

# Main loop.
start = time.perf_counter()
for i in range(poseNumber):

    print("Processing frame number {}...\n".format(i+1))
    img1 = cv2.imread(imgPath+study+'\\'+'cam769_pos{}.pgm'.format(i),0)
    img2 = cv2.imread(imgPath+study+'\\'+'cam802_pos{}.pgm'.format(i),0)
    (key1, des1) = featureType.detectAndCompute(img1,None)
    (key2, des2) = featureType.detectAndCompute(img2,None)
    # Dependent on type of feature. SIFT has length 128, SURF has 64.
    desLen = des1.shape[1]
    
    # Assembling descriptors of form [u,v,[descriptor]] for the current frame, 
    # where (u,v) is the pixel coordinates of the keypoint.
    c1des = np.zeros((len(key1),(desLen+2)),dtype=np.float32)
    c2des = np.zeros((len(key2),(desLen+2)),dtype=np.float32)
    c1des[:,:2] = np.array([key1[j].pt for j in range(len(key1))])
    c1des[:,2:] = des1
    c2des[:,:2] = np.array([key2[j].pt for j in range(len(key2))])
    c2des[:,2:] = des2

    # Correct for distortion.
    c1des[:,:2] = cg.correct_dist(c1des[:,:2],fc1,pp1,kk1,kp1)
    c2des[:,:2] = cg.correct_dist(c2des[:,:2],fc2,pp2,kk2,kp2)

    # Create a list of 'DMatch' objects, which can be queried to obtain
    # matched keypoint indices and their spatial positions.
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
    frameDes = np.ones((len(inepi),(4+desLen)),dtype=np.float32)

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

    elif estMethod == 1:
        frameIdx, dbIdx = lm.dbmatch3D(frameDes,db,beta2)
        # Horn's method needs at least 3 points in each frame.
        if (len(frameIdx) >= 3 and len(dbIdx) >= 3):
            frameMatched = frameDes[frameIdx]
            dbMatched = db[dbIdx]
        else: 
            print("Not enough matches with database, returning previous pose.\n")
            poseList[i,:] = pEst
            continue
        
    elif estMethod == 2:
        indb1, dbm1, indb2, dbm2 = lm.dbmatch(c1des[:,2:],c2des[:,2:],
                                                  db[:,4:],beta2)
	
    # Estimate pose. Points in frameDes that are not matched with landmarks in
    # the database are added to database.
    if i != 0:
        
        if estMethod == 1:
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
            n_est = frameMatched.shape[0]
            estList[i] = n_est
            
            # Add new entries to database:
            frameNew = np.delete(frameDes,[frameIdx],axis=0)
            frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
            db = np.append(db,frameNew,axis=0)
        
        if estMethod == 2:
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
                pflag == -1
                n_est = 0
                print("No matches with database, returning previous pose.\n")
            H = cg.vec2mat(pEst)
            
            # Record number of points used to estimate pose.
            estList[i] = n_est
            
            lmInd = []
            # Add new entries to database:
            if pflag == 0:
                for j in range(len(in1)):
                    if (in1[j] not in indb1) and (in2[j] not in indb2):
                        lmInd.append(j)
                frameNew = frameDes[lmInd,:]
                frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
                db = np.append(db,frameNew,axis=0)

    print("Pose estimate for frame {} is:\n {} \n".format((i+1),pEst))
    poseList[i,:] = pEst

    nlm = db.shape[0]
    lmList[i] = nlm
    
    print("{} landmarks in database.\n".format(nlm))

end = time.perf_counter()
timeTaken = end - start
print("Time taken: {}.".format(timeTaken))

# Save arrays of number of landmarks, and number of data points used to
# estimate pose.
np.savez(r'Results/info_{}_{}_{}'.format(study,sys.argv[2],estMethod),
         lmList=lmList,estList=estList)

Header  = ("Feature Type: {} \nStudy: {} \nPose estimation method: {}" + 
           "\nIntra-frame matching beta: {} \nDatabase matching beta: {}\n")
Footer  = "\n{} total landmarks in database.\nTime taken: {}."
outPath = (r"Results\Poses_{}_{}.txt")
	  
np.savetxt(outPath.format(study,sys.argv[2]),poseList,
           header=Header.format(sys.argv[2],study,estMethod,beta1,beta2),
           footer=Footer.format(nlm,timeTaken))