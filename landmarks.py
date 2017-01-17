"""
landmarks.py
Functions used in markerless tracking.

This module contains:
    dbmatch: Database matching for 2 lists of 3D points.
    objective_function1: Calculates sum(|X'-H*X|).
    pose_estimation1: Iteratively finds pose by minimizing objective_function1. Uses
                      Nelder-Mead algorithm.
"""
import numpy as np
import cv2
from scipy.optimize import minimize
import camerageometry as cg

def dbmatch(frameDes,db,beta2,flag=1):
# Database matching.
# Note: Different features such as SURF features may have different descriptor lengths.
#       As long as the first 4 elements of the descriptor are the homogenized coordinates,
#       the rest of the descriptor can be of any length. 
# Inputs:
#     frameDes - Nx128 (for SIFT) array of descriptors for landmarks found in current frame.
#     db       - Mx128 (for SIFT) array of descriptors for landmarks stored in database.
#     beta2    - Parameter used for nearest neighbour matching (SIFT and SURF only).
#     flag     - Depending on the type of descriptor, may be 1 or 2.
# Outputs:
#     frameIdx - array of indices for the frame descriptor database.
#     dbIdx    - array of indices from database.

    matchList = []
    frameIdx = []
    dbIdx = []

    if flag == 1:

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(frameDes[:,4:],db[:,4:],k=2)

        for m, n in matches:

            if m.distance < beta2*n.distance:

                frameIdx.append(m.queryIdx)
                dbIdx.append(m.trainIdx)

    elif flag == 2:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frameDes[:,4:],db[:,4:])

        for m in matches:

            frameIdx.append(m.queryIdx)
            dbIdx.append(m.trainIdx)

    frameIdx = np.array(frameIdx)
    dbIdx = np.array(dbIdx)

    return frameIdx, dbIdx

def objective_function1(pEst,Xframe,Xdb):
# Objective function to be minimised to estimate pose.
# Inputs:
#     pEst   - Current pose estimate (six vector).
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions in current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions in database.
# Outputs:
#     ret    - the summed norms of |Xframe - H*Xdb|, H being the 4x4 representation of pEst.

    H = cg.vec2mat(pEst[0],pEst[1],pEst[2],pEst[3],pEst[4],pEst[5])
    diffX = abs(Xframe.T - mdot(H,Xdb.T))
    vec = np.apply_along_axis(np.linalg.norm,0,diffX)
    ret = sum(vec)

    return ret

def pose_estimation1(pEst,Xframe,Xdb):
# Estimates pose by minimising objective_function1 using iterations of the Nelder-Mead algorithm.
# Inputs:
#     pEst   - Initial pose estimate (six vector).
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions in current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions in database.
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