"""
camerageometry.py
Functions used in markerless tracking.

This module contains:
    mdot: Matrix multiplication of >= 2 arrays.
    correct_dist: Corrects for distortion, translated from Andre's IDL code.
    linear_triangluation: Linear triangulation for a stereo setup.
    skew: Skew symmetric matrix from a 3D vector.
    epipolar_constraint: Accepts or rejects matches based on epipolar geometry.
    rectify_fusiello: Performs stereo rectification, translated from Andre's IDL code.
    generate_dds: Outputs centering values for use in rectifyfusiello.
    vec2mat: converts six vector to 4x4 matrix.
    mat2vec: converts 4x4 matrix to six vector.
    hornmm: Horn's least squares solution for absolute orientation.
"""
import numpy as np
import cv2

def mdot(*args):
# Matrix multiplication for more than 2 arrays, as np.mdot() can only take 2 arguments.

    ret = args[0]

    for a in args[1:]:

        ret = np.dot(ret,a)

    return ret
  
def correct_dist(vec,fc,c,k,p): 
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
 
def linear_triangulation(P1,P2,x1,x2):
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

        print("Error in skew(): input vector must have 3 elements!")
        return

    else:

        skew = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])

    return skew
 
def epipolar_constraint(locs1,locs2,T1,T2):
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
    successfulMatches = np.where(abs(tlocsy1-tlocsy2)<=5)[0]

    return successfulMatches
 
def rectify_fusiello(P1,P2,d1=np.zeros(2),d2=np.zeros(2)):
# Translation of Andre's IDL function rectify_fusiello.
# A[R|t] factorisation is done by the opencv routine cv2.decomposeProjectionMatrix()
# Inputs:
#     P1 & P2       - 3x4 camera projection matrices for cameras 1 and 2.
#     d1 & d2       - Centering parameters obtained from generatedds, defaulted to 0.
# Outputs:
#     Prec1 & Prec2 - Rectified camera projection matrices.
#     Tr1 & Tr2     - Rectifying transforms applied to homogeneous pixel coordinates.

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

def generate_dds(Tr1,Tr2):
# Generates DD1 and DD2 for centering, used in conjunction with rectifyfusiello.

    p = np.array([640,480,1],ndmin=2).T
    px = np.dot(Tr1,p)
    DD1 = (p[:2]-px[:2]) / px[2]
    px = np.dot(Tr2,p)
    DD2 = (p[:2]-px[:2]) / px[2]
    DD1[1] = DD2[1]

    return DD1, DD2

def vec2mat(*args):
# Converts a six vector represenation of motion to a 4x4 matrix.
# Assumes yaw, pitch, roll are in degrees.
# Inputs:
#     *args - either 6 numbers (yaw,pitch,roll,x,y,z) or an array with 6 elements.
# Outputs:
#     t     - 4x4 matrix representation of six vector.

    if len(args) == 6:
    
        yaw = args[0]
        pitch = args[1]
        roll = args[2]
        x = args[3]
        y = args[4]
        z = args[5]

    elif len(args) == 1:

        try:
            l = len(args[0])
            if l != 6:
                print("Error in vec2mat(): input must be 6 element array or 6 numbers!")
                return

        except:
            print("Error in vec2mat(): input must be 6 element array or 6 numbers!")
            return
		
        yaw = args[0][0]
        pitch = args[0][1]
        roll = args[0][2]
        x = args[0][3]
        y = args[0][4]
        z = args[0][5]

    else:

        print("Error in vec2mat(): input must be 6 element array or 6 numbers!")
        return
	
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

def mat2vec(H):
# Converts a 4x4 representation of pose to a 6 vector.
# Inputs:
#     H - 4x4 matrix.
# Outputs:
#     v - [yaw,pitch,roll,x,y,z] (yaw,pitch,roll are in degrees)

    if np.array(H).shape != np.eye(4).shape:
        print("Error in mat2vec(): Input array must be 4x4!")
        return

    sy = -H[2,0]
    cy = 1-(sy*sy)

    if cy > 0.00001:
        cy = np.sqrt(cy)
        cx = H[2,2]/cy
        sx = H[2,1]/cy
        cz = H[0,0]/cy
        sz = H[1,0]/cy
    else:
        cy = 0.0
        cx = H[1,1]
        sx = -H[1,2]
        cz = 1.0
        sz = 0.0

    r2deg = (180/np.pi)
    v = np.array([np.arctan2(sx,cx)*r2deg,np.arctan2(sy,cy)*r2deg,np.arctan2(sz,cz)*r2deg,
                  H[0,3],H[1,3],H[2,3]])

    return v

def hornmm(Xframe,Xdb):
# Translated from hornmm.pro
# Least squares solution to X' = H*X. Here, X' is Xframe, X is Xdb.
# Inputs:
#     Xframe - Nx4 array of triangulated, homogeneous 3D landmark positions in current frame.
#     Xdb    - Nx4 array of triangulated, homogeneous 3D landmark positions in database.
# Outputs:
#     H      - Transformation between X and X', or the new pose.
#
# Implements method in "Closed-form solution of absolute orientation using unit quaternions",
# Horn B.K.P, J Opt Soc Am A 4(4):629-642, April 1987.

    N = Xdb.shape[0]
    
    xc  = np.sum(Xdb[:,0])/N
    yc  = np.sum(Xdb[:,1])/N
    zc  = np.sum(Xdb[:,2])/N	
    xfc = np.sum(Xframe[:,0])/N
    yfc = np.sum(Xframe[:,1])/N 	
    zfc = np.sum(Xframe[:,2])/N

    xn  = Xdb[:,0] - xc
    yn  = Xdb[:,1] - yc
    zn  = Xdb[:,2] - zc
    xfn = Xframe[:,0] - xfc
    yfn = Xframe[:,1] - yfc
    zfn = Xframe[:,2] - zfc

    sxx = np.dot(xn,xfn)
    sxy = np.dot(xn,yfn)
    sxz = np.dot(xn,zfn)
    syx = np.dot(yn,xfn)
    syy = np.dot(yn,yfn)
    syz = np.dot(yn,zfn)
    szx = np.dot(zn,xfn)
    szy = np.dot(zn,yfn)
    szz = np.dot(zn,zfn)

    M = np.array([[sxx,syy,sxz],
                  [syx,syy,syz],
                  [szx,szy,szz]])

    N = np.array([[(sxx+syy+szz),(syz-szy),(szx-sxz),(sxy-syx)],
                  [(syz-szy),(sxx-syy-szz),(sxy+syx),(szx+sxz)],
                  [(szx-sxz),(sxy+syx),(-sxx+syy-szz),(syz+szy)],
                  [(sxy-syx),(szx+sxz),(syz+szy),(-sxx-syy+szz)]])

    eVal,eVec = np.linalg.eig(N)
    index = np.argmax(eVal)
    vec = eVec[:,index]
    q0 = vec[0]
    qx = vec[1]
    qy = vec[2]
    qz = vec[3]
	
    X = np.array([[(q0*q0+qx*qx-qy*qy-qz*qz),2*(qx*qy-q0*qz),2*(qx*qz+q0*qy),0],
                  [2*(qy*qx+q0*qz),(q0*q0-qx*qx+qy*qy-qz*qz),2*(qy*qz-q0*qx),0],
                  [2*(qz*qx-q0*qy),2*(qz*qy+q0*qx),(q0*q0-qx*qx-qy*qy+qz*qz),0],
                  [0,0,0,1]])

    Xpos = np.array([xc,yc,zc,1])
    Xfpos = np.array([xfc,yfc,zfc,1])
    d = Xpos - np.dot(np.linalg.inv(X),Xfpos)

    Tr = np.array([[1,0,0,-d[0]],
                   [0,1,0,-d[1]],
                   [0,0,1,-d[2]],
                   [0,0,0,1]])

    H = np.dot(X,Tr)

    return H