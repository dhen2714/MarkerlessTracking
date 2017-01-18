"""

Estimates pose using feature detectors.

For each frame:
 - Detect keypoints for both camera images.
 - Match keypoints within the frame.
 - Correct for distortion.
 - Triangulate.
 - Compare triangulated landmarks with those in the database.
 - Pose is estimated using Horn's method.

Pose estimation here is different to the implementation in Andre's IDL code. Pose is estimated by finding the transformation H that 
minimises |X' - H*X|, where X' is the triangulated position of an observed landmark in the current frame, and X is the position
of the landmark in the database. Features must thus be detected by both cameras in the stereo setup to be used in pose estimation,
which is not the case in Andre's implementation.

For functions used, robotexp.py, camerageometry.py and landmarks.py

"""
import sys
import numpy as np
import cv2
import camerageometry as cg
import landmarks as lm
import robotexp
import time

start = time.clock()

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

# Name of study, featureType from user input. E.g., 'yidi_nostamp' 'sift'
study, featureType = robotexp.handle_args(sys.argv)
print("\nChosen study is: {}\nChosen feature type is: {}\n\n".format(study,sys.argv[2].lower())) 

imgPath = "C:\\Users\\dhen2714\\Documents\\PHD_Thesis\\Experiments\\YidiRobotExp\\robot_experiment\\images\\"

poseNumber = 30 # Number of frames to process.
poseList = np.zeros((poseNumber,6))

bf = cv2.BFMatcher()
beta1 = 0.6 # NN matching parameter for intra-frame matching.
beta2 = 0.6 # For database matching.

# Rectify for outlier removal with epipolar constraint.
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2)
DD1,DD2 = cg.generate_dds(Tr1,Tr2)
Prec1,Prec2,Tr1,Tr2 = cg.rectify_fusiello(P1,P2,DD1,DD2)

# Main loop.
for i in range(poseNumber):

    print("Processing frame number {}...\n".format(i+1))
    img1 = cv2.imread(imgPath+study+'\\'+'cam769_pos{}.pgm'.format(i),0)
    img2 = cv2.imread(imgPath+study+'\\'+'cam802_pos{}.pgm'.format(i),0)
    (key1, des1) = featureType.detectAndCompute(img1,None)
    (key2, des2) = featureType.detectAndCompute(img2,None)
    desLen = des1.shape[1] # Dependent on type of feature. SIFT has length 128, SURF has 64.

    # Create a list of 'DMatch' objects, which can be queried to obtain matched keypoint indices and their spatial positions.
    match = bf.knnMatch(des1,des2,k=2)
    matchProper = []
    
    for m, n in match:
    
        if m.distance < beta1*n.distance:
        
            matchProper.append(m)

    matchProper = np.array(matchProper) # Convert to numpy array for compatability.
	
    # f1Points and f2Points are Nx2 pixel coordinate arrays of keypoints matched within the frame, where N is the number of matches
    # between the two cameras in the current frame.
    f1Points = np.array([key1[matchProper[j].queryIdx].pt for j in range(len(matchProper))])
    f2Points = np.array([key2[matchProper[j].trainIdx].pt for j in range(len(matchProper))])

    # Correct for distortion.
    f1Points = cg.correct_dist(f1Points,fc1,pp1,kk1,kp1)
    f2Points = cg.correct_dist(f2Points,fc2,pp2,kk2,kp2)

    # Verify the matches with the epipolar constraint. indices is a 1D np array, containing the indices of points in f1Points and
    # f2Points which satisfy the epipolar constraint.
    indices = cg.epipolar_constraint(f1Points,f2Points,Tr1,Tr2)
 
    # Triangulate verified points.
    X = cg.linear_triangulation(P1,P2,f1Points[indices],f2Points[indices])
    
    # Create an array of descriptors of form [x,y,z,1,[descriptor]] triangulated points in the current frame.
    frameDes = np.ones((len(indices),(4+desLen)),dtype=np.float32)

    for j in range(len(indices)):

        ind = indices[j]
        frameDes[j,:3] = X[j,:3]
        frameDes[j,4:] = (des1[matchProper[ind].queryIdx] + des2[matchProper[ind].trainIdx])/2

    # Database matching. If it is the first frame, the database is a copy of frameDes.
    if i == 0:

        db = np.copy(frameDes)
        frameMatched = frameDes
        dbMatched = db
        pest = [0,0,0,0,0,0]

    else:

        frameIdx,dbIdx = lm.dbmatch(frameDes,db,beta2)
        frameMatched = frameDes[frameIdx]
        dbMatched = db[dbIdx]

    # Estimate pose. Points in frameDes that are not matched with landmarks in the database are added to database.
    if i != 0:

        H = cg.hornmm(frameMatched[:,:4],dbMatched[:,:4])
        pest = cg.mat2vec(H)
        print("Pose estimate for frame {} is:\n {} \n".format((i+1),pest))
        # Add new entries to database:
        frameNew = np.delete(frameDes,[frameIdx],axis=0)
        frameNew[:,:4] = cg.mdot(np.linalg.inv(H),frameNew[:,:4].T).T
        db = np.append(db,frameNew,axis=0)

    poseList[i,:] = pest

    print("{} landmarks in database.\n".format(db.shape[0]))

timeTaken = time.clock() - start
print("Time taken: {} seconds".format(timeTaken))

Header = "Feature Type: {} \nStudy: {} \nIntra-frame matching beta: {} \nDatabase matching beta: {}\n"
Footer = "\n{} total landmarks in database.\nTime taken: {} seconds."
	  
np.savetxt('Results\Test.txt',poseList,
           header=Header.format(sys.argv[2],study,beta1,beta2),
           footer=Footer.format(db.shape[0],timeTaken))