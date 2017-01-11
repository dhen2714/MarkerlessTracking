"""

Functions that are used in sift_motion_main.py

- mdot: matrix multiplication of >= 2 arrays.
- correctdist: corrects for distortion, translated from Andre's IDL code.
- lineartriangluation: linear triangulation for a stereo setup.
- skew: skew symmetric matrix from a 3D vector.
- epipolarconstraint: accepts or rejects matches based on epipolar geometry.
- rectifyfusiello: performs stereo rectification, translated from Andre's IDL code.
- generatedds: outputs centering values for use in rectifyfusiello.
- vec2mat: converts six vector to 4x4 matrix.
- objective_function1: calculates objective function to be minimised with pose_estimation1.
- pose_estimation1: estimates pose by minimising objective_function1, uses Nelder-Mead algorithm.

"""
import numpy as np
import cv2
from scipy.optimize import minimize

def mdot(*args):
# Matrix multiplication for more than 2 arrays, as np.mdot() can only take 2 arguments.

    ret = args[0]

    for a in args[1:]:

        ret = np.dot(ret,a)

    return ret
  
def correctdist(vec,fc,c,k,p): 
# From Andre's IDL code.
# Inputs:
#     vec - Nx2 array of distorted pixel coordinates.
#     fc  - 2 element focal length.
#     c   - principal point.
#     k   - 2 radial distortion coefficients.
#     p   - 2 tangential distortion coefficients.
# Outputs:
#     cvec - Nx2 array of distortion-corrected pixel coordinates.

    ud = vec[:,0]
    vd = vec[:,1]
    xn = (ud - c[0])/fc[0] # Normalise points
    yn = (vd - c[1])/fc[1]
    x = xn
    y = yn
 
    for i in range(19):

        r2 = x*x + y*y
        r4 = r2*r2
        k_radial = 1 + k[0]*r2 + k[1]*r4
        delta_x = 2*p[0]*x*y + p[1]*(r2 + 2*x*x)
        delta_y = 2*p[1]*x*y + p[0]*(r2 + 2*y*y)
        x = (xn - delta_x)/k_radial
        y = (yn - delta_y)/k_radial
 
    x = fc[0]*x + c[0] # Undo normalisation
    y = fc[1]*y + c[1]
    cvec = np.array([x,y]).T

    return cvec
 
def lineartriangulation(P1,P2,x1,x2):
# Linear triangulation method.
# Inputs:
#     P1 & P2 - Camera projection matrices for cameras 1 and 2.
#     x1 & x2 - Nx2 arrays of pixel coordinates corresponding to matched keypoints.
# Outputs:
#     X       - Nx3 array of triangulated points.

    N = x1.shape[0]
    X = np.zeros((N,3),dtype=float)

    for i in range(N):

        hx1 = np.concatenate((x1[i,:],[1])) # Homogenize pixel coordinates.
        hx2 = np.concatenate((x2[i,:],[1]))
        skewx1 = skew(hx1)
        skewx2 = skew(hx2)
        A1 = np.matmul(skewx1,P1)
        A2 = np.matmul(skewx2,P2)
        A = np.concatenate((A1,A2),axis=0)
        U,S,Vt = np.linalg.svd(A)
        V = Vt.T # np.linalg.svd() returns U,S,V'
        X[i,:] = V[:3,-1]/V[-1,-1] # Divide homogeneous vector by last element

    return X

def skew(u):
# Outputs a skew-symmetric matrix for an input vector u.

    if len(u) != 3:

        print("Error in skew: input vector must have 3 elements!")
        quit()

    else:

        skew = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])

    return skew
 
def epipolarconstraint(locs1,locs2,T1,T2):
# Verify matches based on epipolar geometry.
# Inputs:
#     locs1 & locs2     - Nx2 arrays of keypoint pixel coordinates.
#     T1 & T2           - Rectifying transforms obtained from rectifyfusiello.
# Outputs:
#     successfulMatches - Array of indices corresponding to keypoints which satisfied constraints.

    N = locs1.shape[0]
    locs1 = np.concatenate((locs1,np.ones((N,1))),1)
    locs2 = np.concatenate((locs2,np.ones((N,1))),1)
    tlocs1 = np.matmul(T1,locs1.T)
    tlocs2 = np.matmul(T2,locs2.T)
    tlocsy1 = tlocs1[1,:]/tlocs1[2,:]
    tlocsy2 = tlocs2[1,:]/tlocs2[2,:]
    successfulMatches = np.where(abs(tlocsy1-tlocsy2)<=1)[0]

    return successfulMatches
 
def rectifyfusiello(P1,P2,d1=np.zeros(2),d2=np.zeros(2)):
# Translation of Andre's IDL function rectify_fusiello.
# A[R|t] factorisation is done by the opencv routine cv2.decomposeProjectionMatrix()
# Inputs:
#     P1 & P2       - 3x4 camera projection matrices for cameras 1 and 2.
#     d1 & d2       - Centering parameters obtained from generatedds, defaulted to 0.
# Outputs:
#     Tr1 & Tr2     - Rectifying transforms applied to homogeneous pixel coordinates.
#     Prec1 & Prec2 - Rectified camera projection matrices.

    K1,R1,C1,_,_,_,_ = cv2.decomposeProjectionMatrix(P1)
    K2,R2,C2,_,_,_,_ = cv2.decomposeProjectionMatrix(P2)
    C1 = cv2.convertPointsFromHomogeneous(C1.T).reshape(3,1)
    C2 = cv2.convertPointsFromHomogeneous(C2.T).reshape(3,1)

    oc1 = mdot(-R1.T,np.linalg.inv(K1),P1[:,3])
    oc2 = mdot(-R2.T,np.linalg.inv(K2),P2[:,3])
 
    v1 = (oc2-oc1).T
    v2 = np.cross(R1[2,:],v1)
    v3 = np.cross(v1,v2)
 
    R = np.array([v1/np.linalg.norm(v1),v2/np.linalg.norm(v2),v3/np.linalg.norm(v3)]).reshape(3,3)
 
    Kn1 = np.copy(K2)
    Kn1[0,1] = 0
    Kn2 = np.copy(K2)
    Kn2[0,1] = 0
 
    Kn1[0,2] = Kn1[0,2] + d1[0]
    Kn1[1,2] = Kn1[1,2] + d1[1]
    Kn2[0,2] = Kn2[0,2] + d2[0]
    Kn2[1,2] = Kn2[1,2] + d2[1]
 
    t1 = np.matmul(-R,C1)
    t2 = np.matmul(-R,C2)
    Rt1 = np.concatenate((R,t1),1)
    Rt2 = np.concatenate((R,t2),1)
    Prec1 = np.dot(Kn1,Rt1)
    Prec2 = np.dot(Kn2,Rt2)
 
    Tr1 = np.dot(Prec1[:3,:3],np.linalg.inv(P1[:3,:3]))
    Tr2 = np.dot(Prec2[:3,:3],np.linalg.inv(P2[:3,:3]))
 
    return Prec1,Prec2,Tr1,Tr2

def generatedds(Tr1,Tr2):
# Generates DD1 and DD2 for centering, used in conjunction with rectifyfusiello.

    p = np.array([640,480,1],ndmin=2).T
    px = np.dot(Tr1,p)
    DD1 = (p[:2]-px[:2]) / px[2]
    px = np.dot(Tr2,p)
    DD2 = (p[:2]-px[:2]) / px[2]
    DD1[1] = DD2[1]

    return DD1, DD2

def dbmatch(frameDes,db,beta2):
# Database matching.
# Inputs:
#     frameDes - Nx128 array of descriptors for landmarks found in current frame.
#     db       - Mx128 array of descriptors for landmarks stored in database.
#     beta2    - Parameter used for nearest neighbour matching.
# Outputs:
#     frameIdx - array of indices for the frame descriptor database.
#     dbIdx    - array of indices from database.

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(frameDes[:,4:],db[:,4:],k=2)
    matchList = []
    frameIdx = []
    dbIdx = []

    for m, n in matches:

        if m.distance < beta2*n.distance:

            frameIdx.append(m.queryIdx)
            dbIdx.append(m.trainIdx)

    frameIdx = np.array(frameIdx)
    dbIdx = np.array(dbIdx)

    return frameIdx, dbIdx

def vec2mat(yaw,pitch,roll,x,y,z):
# Converts a six vector represenation of motion to a 4x4 matrix.
# Assumes yaw, pitch, roll are in degrees.

    ax = (np.pi/180)*yaw
    ay = (np.pi/180)*pitch
    az = (np.pi/180)*roll
    t1 = np.array([[1,0,0,0],
                   [0,np.cos(ax),-np.sin(ax),0],
                   [0,np.sin(ax),np.cos(ax),0],
                   [0,0,0,1]])
    t2 = np.array([[np.cos(ay),0,np.sin(ay),0],
                   [0,1,0,0],
                   [-np.sin(ay),0,np.cos(ay),0],
                   [0,0,0,1]])
    t3 = np.array([[np.cos(az),-np.sin(az),0,0],
                   [np.sin(az),np.cos(az),0,0],
                   [0,0,1,0],
                   [0,0,0,1]])
    tr = np.array([[1,0,0,x],
                   [0,1,0,y],
                   [0,0,1,z],
                   [0,0,0,1]])

    t = mdot(tr,t3,t2,t1)

    return t

def objective_function1(pEst,Xframe,Xdb):
# Objective function to be minimised to estimate pose.
# Inputs:
#     pEst   - Current pose estimate (six vector).
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions in current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions in database.
# Outputs:
#     ret    - the summed norms of |Xframe - H*Xdb|, H being the 4x4 representation of pEst.

    H = vec2mat(pEst[0],pEst[1],pEst[2],pEst[3],pEst[4],pEst[5])
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