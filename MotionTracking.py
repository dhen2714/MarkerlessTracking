"""
Estimates pose using feature detectors.

Accepted arguments are the study type and the feature type, E.g:

python MotionTracking.py yidi_nostamp sift

For each frame:
 - Detect keypoints for both camera images.
 - Match keypoints within the frame.
 - Correct for distortion.
 - Triangulate.
 - Compare triangulated landmarks with those in the database.
 - Pose is estimated using Horn's method.

Pose estimation here is different to the implementation in Andre's IDL code. 
Pose is estimated by finding the transformation H that minimises |X' - H*X|, 
where X' is the triangulated position of an observed landmark in the current 
frame, and X is the position of the landmark in the database. Features must 
thus be detected by both cameras in the stereo setup to be used in pose 
estimation, which is not the case in Andre's implementation.

For functions used, see robotexp.py, camerageometry.py and landmarks.py

"""
import sys
import numpy as np
import cv2
import camerageometry as cg
import landmarks as lm
import robotexp
import time

imgPath = ("C:\\Users\\dhen2714\\Documents\\PHD_Thesis\\Experiments" + 
           "\\YidiRobotExp\\robot_experiment\\images\\")

# Load camera matrices.
P = np.fromfile("C:\\Users\\dhen2714\\Documents\\PHD_Thesis\\Experiments\\"+
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

# Name of study, featureType from user input. E.g., 'yidi_nostamp' 'sift'
study, featureType, estMethod = robotexp.handle_args(sys.argv)
print("\nChosen study is: {}\n\nChosen feature type is: {}\n\n"
      .format(study,sys.argv[2].lower()))
input("Press ENTER to continue.\n\n")

# Initialize matcher, brute force matcher in this case.
bf = cv2.BFMatcher()
beta1 = 0.6 # NN matching parameter for intra-frame matching.
beta2 = 0.6 # For database matching.

# Rectify for outlier removal with epipolar constraint.
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2)
DD1,DD2 = cg.generate_dds(Tr1,Tr2)
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2,DD1,DD2)

start = time.clock()
# Main loop.
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
    c2des[:,:2] = cg.correct_dist(c2des[:,:2],fc1,pp1,kk1,kp1)

    # Create a list of 'DMatch' objects, which can be queried to obtain
    # matched keypoint indices and their spatial positions.
    match = bf.knnMatch(des1,des2,k=2)
    matchProper = []
    
    for m, n in match:
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
        frameMatched = frameDes[frameIdx]
        dbMatched = db[dbIdx]
        
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
            
            # Add new entries to database:
            frameNew = np.delete(frameDes,[frameIdx],axis=0)
            frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
            db = np.append(db,frameNew,axis=0)
        
        if estMethod == 2:
            pEst, pflag = lm.GN_estimation(P1,P2,c1des[indb1,:2],c2des[indb2,:2],
                                    db[dbm1,:4],db[dbm2,:4],pEst)
            H = cg.vec2mat(pEst)
            
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

    print("{} landmarks in database.\n".format(db.shape[0]))

timeTaken = time.clock() - start
print("Time taken: {} seconds".format(timeTaken))

Header  = ("Feature Type: {} \nStudy: {} \n Pose estimation method: {}" + 
           "\nIntra-frame matching beta: {} \nDatabase matching beta: {}\n")
Footer  = "\n{} total landmarks in database.\nTime taken: {} seconds."
outPath = (r"Results\Poses_{}_{}.txt")
	  
np.savetxt(outPath.format(study,sys.argv[2]),poseList,
           header=Header.format(sys.argv[2],study,beta1,beta2),
           footer=Footer.format(db.shape[0],timeTaken))