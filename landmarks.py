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

def dbmatch(frameDes,db,beta2,flag=1):
# Database matching.
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