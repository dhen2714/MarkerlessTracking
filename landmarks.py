"""
landmarks.py
Functions used in markerless tracking.

This module contains:
    dbmatch: Database matching for 2 lists of 3D points.
    detect_outliers: Returns an array of indices of outliers.
    remove_duplicates: Removes duplicate matches.
    objective_function1: Calculates sum(|X'-H*X|).
    pose_estimation1: Iteratively finds pose by minimizing 
                      objective_function1. Uses Nelder-Mead algorithm.
"""
import numpy as np
import cv2
from scipy.optimize import minimize
import camerageometry as cg

def dbmatch(des1,des2,db,beta2):

    matches1 = []
    matches2 = []
    bf = cv2.BFMatcher()
    
    matches = bf.knnMatch(des1,db,k=2)
    for m, n in matches:
        if m.distance < beta2*n.distance:
            matches1.append(m)
    
    matches = bf.knnMatch(des2,db,k=2)
    for m, n in matches:
        if m.distance < beta2*n.distance:
            matches2.append(m)
            
    matches1 = remove_duplicates(np.array(matches1))
    matches2 = remove_duplicates(np.array(matches2))
    
    indb1 = np.array([matches1[j].queryIdx for j in range(len(matches1))])
    indb2 = np.array([matches2[j].queryIdx for j in range(len(matches2))])
    dbm1 = np.array([matches1[j].trainIdx for j in range(len(matches1))])
    dbm2 = np.array([matches2[j].trainIdx for j in range(len(matches2))])
    
    return indb1, dbm1, indb2, dbm2

def dbmatch3D(frameDes,db,beta2,flag=1):
# Database matching for feature points that have been triangulated before
# matching.
# Note: Different features such as SURF features may have different descriptor
#       lengths.
#       As long as the first 4 elements of the descriptor are the homogenized 
#       coordinates,the rest of the descriptor can be of any length. 
# Inputs:
#     frameDes - Nx128 (for SIFT) array of descriptors for landmarks found in 
#                current frame.
#     db       - Mx128 (for SIFT) array of descriptors for landmarks stored in 
#                database.
#     beta2    - Parameter used for nearest neighbour matching (SIFT and SURF 
#                only).
#     flag     - Depending on the type of descriptor, may be 1 or 2.
# Outputs:
#     frameIdx - array of indices for the frame descriptor database.
#     dbIdx    - array of indices from database.

    matchProper = []
    frameIdx = []
    dbIdx = []

    if flag == 1:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(frameDes[:,4:],db[:,4:],k=2)

        for m, n in matches:
            if m.distance < beta2*n.distance:
                matchProper.append(m)

    elif flag == 2:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(frameDes[:,4:],db[:,4:],k=2)

        for m, n in matches:
            if m.distance < beta2*n.distance:
                matchProper.append(m)

    matchProper = remove_duplicates(np.array(matchProper))
    frameIdx = np.array([matchProper[j].queryIdx 
                         for j in range(len(matchProper))])
    dbIdx = np.array([matchProper[j].trainIdx 
                      for j in range(len(matchProper))])

    return frameIdx, dbIdx
	
def detect_outliers(array):
# Removes outliers based on modified Z-score. Translated from Andre's IDL code.
# Inputs:
#     array    - 1D array.
# Outputs:
#     outliers - Array indices of outliers.

    med = np.median(array)
    mad = np.median(abs(array - med))
    
    if mad == 0:
        return np.array(0)
    
    mi = abs(0.6745*(array - med))/mad

    outliers = np.where(mi > 3.5)[0]

    return outliers
	
def remove_duplicates(matches):
# Removes duplicate matches from a list of match objects.
# Inputs:
#     matches       - Numpy array of DMatch objects.
# Outputs:
#     matchesUnique - Same as input, but with duplicate matches removed.

    matchIndices = np.array([(matches[j].queryIdx,matches[j].trainIdx)
                             for j in range(len(matches))])
							  
    _,countsT = np.unique(matchIndices[:,1],return_counts=True)

    matchesUnique = matches[np.where(countsT==1)[0]]

    return matchesUnique

def euler_jacobian(P,H,X,x_e):
    
    n = X.shape[0]
    J = np.ones((2*n,6))

    for i in range(6):
        h_i = cg.mat2vec(H)
        h_i[i] = h_i[i] + 0.5
        H_p = cg.vec2mat(h_i)
        x_p = cg.mdot(P,H_p,X.T)
        x_p = np.apply_along_axis(lambda v: v/v[-1],0,x_p)
        
        J[:,i]  = ((x_p - x_e)/0.5)[:2,:].flatten(order='F')
        
    return J
    
def GN_estimation(P1,P2,uv1m,uv2m,dbc1,dbc2,pose,iter_n=10,olThold=2):
# Estimates pose using a Gauss-Newton iterative algorithm. Based on Andre's
# IDL implementation numerical_estimation_2cams_v2.pro
# Inputs:
#     uv1m    - N1x2 array of pixel coordinates of features that have been
#               matched to landmarks in database for camera 1.
#     dbc1    - N1x4 array of landmarks in the database that have been matched
#               to features in camera 1 [X Y Z 1].
#     uv2m    - N2x2 array of pixel coordinates of features that have been
#               matched to landmarks in database for camera 2.
#     dbc2    - N2x4 array of landmarks in the database that have been matched
#               to features in camera 1 [X Y Z 1].
#     pose    - Inital pose estimate. Form [yaw,pitch,roll,x,y,z].
# Outputs:
#     posEst  - Final pose estimate. Form [yaw,pitch,roll,x,y,z].
#     flag    - (-1) if pose could not be estimated due to insufficient (<3)
#               points, 0 otherwise.
# Options:
#     iter_n  - Number of iterations to perform, default is 10 iterations.
#     olThold - Absolute threshold for outlier rejection in pixels. This
#               threshold is used after 4 iterations. Default is 2 pixels.

    uv1m = uv1m.T # Arrays are 2xN (not homogenized).
    uv2m = uv2m.T
    n1 = dbc1.shape[0]
    n2 = dbc2.shape[0]
    p_init = pose
    posEst = pose

    for i in range(iter_n):
    
        print("Iteration number",(i+1))
        H_i = cg.vec2mat(posEst)

        if dbc1.size:
            uv1e = cg.mdot(P1,H_i,dbc1.T)
            uv1e = np.apply_along_axis(lambda v: v/v[-1],0,uv1e)
            J1 = euler_jacobian(P1,H_i,dbc1,uv1e)
        
            squerr = np.sum(np.square(uv1e[:2,:] - uv1m),0)
            outliers = detect_outliers(squerr)
        
            if outliers.shape:
                uv1m = np.delete(uv1m,outliers,axis=1)
                uv1e = np.delete(uv1e,outliers,axis=1)
                dbc1 = np.delete(dbc1,outliers,axis=0)
                Joutliers = np.array([2*outliers,(2*outliers+1)]).flatten()
                J1 = np.delete(J1,Joutliers,axis=0)
                n1 = n1 - len(outliers)
            elif i >= 4:
                absOutliers = np.where(squerr > olThold)[0]
                if len(absOutliers) > 0:
                    uv1m = np.delete(uv1m,absOutliers,axis=1)
                    uv1e = np.delete(uv1e,absOutliers,axis=1)
                    dbc1 = np.delete(dbc1,absOutliers,axis=0)
                    Joutliers = np.array([2*absOutliers,(2*absOutliers+1)]).flatten()
                    J1 = np.delete(J1,Joutliers,axis=0)
                    n1 = n1 - len(absOutliers)
        
        if dbc2.size:
            uv2e = cg.mdot(P2,H_i,dbc2.T)
            uv2e = np.apply_along_axis(lambda v: v/v[-1],0,uv2e)
            J2 = euler_jacobian(P2,H_i,dbc2,uv2e)
        
            squerr = np.sum(np.square(uv2e[:2,:] - uv2m),0)
            outliers = detect_outliers(squerr)
        
            if outliers.shape:
                uv2m = np.delete(uv2m,outliers,axis=1)
                uv2e = np.delete(uv2e,outliers,axis=1)
                dbc2 = np.delete(dbc2,outliers,axis=0)
                Joutliers = np.array([2*outliers,(2*outliers+1)]).flatten()
                J2 = np.delete(J2,Joutliers,axis=0)
                n2 = n2 - len(outliers)
            elif i >= 4:
                absOutliers = np.where(squerr > olThold)[0]
                if len(absOutliers) > 0:
                    uv2m = np.delete(uv2m,absOutliers,axis=1)
                    uv2e = np.delete(uv2e,absOutliers,axis=1)
                    dbc2 = np.delete(dbc2,absOutliers,axis=0)
                    Joutliers = np.array([2*absOutliers,(2*absOutliers+1)]).flatten()
                    J2 = np.delete(J2,Joutliers,axis=0)
                    n2 = n2 - len(absOutliers)
                
        if n1 + n2 < 3:
            print("Cannot estimate pose from this frame, return last pose.")
            flag = -1
            return p_init, flag
            
        e1 = (uv1e[:2,:] - uv1m).flatten(order='F')
        e2 = (uv2e[:2,:] - uv2m).flatten(order='F')
        
        if J1.size and J2.size:
            J = np.concatenate((J1,J2),axis=0)
            e = np.concatenate((e1,e2))
        elif J1.size:
            J = J1
            e = e1
        elif J2.size:
            J = J2
            e = e2
        
        A = np.dot(J.T,J)
        b = np.dot(J.T,e)
        
        hcorr = np.linalg.lstsq(A,b)
        posEst = posEst - hcorr[0]
       
    flag = 0
    
    return posEst, flag


def objective_function1(pEst,Xframe,Xdb):
# Objective function to be minimised to estimate pose.
# Inputs:
#     pEst   - Current pose estimate (six vector).
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions in 
#              current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions in 
#              database.
# Outputs:
#     ret    - the summed norms of |Xframe - H*Xdb|, H being the 4x4 
#              representation of pEst.

    H = cg.vec2mat(pEst[0],pEst[1],pEst[2],pEst[3],pEst[4],pEst[5])
    diffX = abs(Xframe.T - mdot(H,Xdb.T))
    vec = np.apply_along_axis(np.linalg.norm,0,diffX)
    ret = sum(vec)

    return ret

def pose_estimation1(pEst,Xframe,Xdb):
# Estimates pose by minimising objective_function1 using iterations of the 
# Nelder-Mead algorithm.
# Inputs:
#     pEst   - Initial pose estimate (six vector).
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions 
#              in current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions 
#              in database.
# Outputs:
#     pOut   - New pose estimate. 

    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':5e-2})
    pest = res.x				   
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':1e-2})
    pest = res.x				   
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':5e-3})
    pest = res.x				   
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':1e-3})
    pest = res.x				   
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':5e-4})
    pest = res.x
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':1e-4})
    pest = res.x
    res = minimize(objective_function1,pEst,args=(Xframe,Xdb),
                   method='nelder-mead',
                   options={'xtol':1e-5,'disp':True})

    pOut = res.x

    return pOut