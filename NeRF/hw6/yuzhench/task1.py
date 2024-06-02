from utils import dehomogenize, homogenize, draw_epipolar
import numpy as np
import cv2
import pdb
import os

from sklearn.preprocessing import StandardScaler



def normalize(points):
        centroid = np.mean(points, axis=0)
        scaled_points = points - centroid

        squared_points = scaled_points ** 2
        sum_of_squares = np.sum(squared_points, axis=1)
        sqrt_sum_of_squares = np.sqrt(sum_of_squares)
        mean_distance = np.mean(sqrt_sum_of_squares)

        const_scale = np.sqrt(2) / mean_distance
        transformation_matrix = np.array([[const_scale, 0, -const_scale * centroid[0]],
                                            [0, const_scale, -const_scale * centroid[1]],
                                            [0, 0, 1]])
        

        ones_row = np.ones((points.shape[0],))
        vstack_result = np.vstack((points.T, ones_row))

        point_after_norm = np.dot(transformation_matrix, vstack_result )
        return point_after_norm[:2].T, transformation_matrix





def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """

    #This will give you an answer you can compare with
    #Your answer should match closely once you've divided by the last entry
    FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    F = np.eye(3)
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################

    norm_pt1, T1 = normalize(pts1)
    norm_pt2, T2 = normalize(pts2)
    # Construct matrix A for the equation Ax = 0
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):

        u1, v1 = norm_pt1[i]
        u2, v2 = norm_pt2[i]
        A[i] = [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]


    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[-1] = 0  # Set smallest singular value to zero
    temp_F = U @ np.diag(S) @ V
    F = T2.T @ temp_F @ T1

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    ###########################################################################
    # TODO: Your code here                                                    #
    ###########################################################################
    # Compute the singular value decomposition of F
    _, _, V = np.linalg.svd(F)

    # Extract the last column of V to get the right singular vector
    e1 = V[-1]

    # Compute the singular value decomposition of F transpose to get the left singular vector
    _, _, V = np.linalg.svd(F.T)
    
    # Extract the last column of V to get the left singular vector
    e2 = V[-1]

    # Normalize the epipoles
    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return e1, e2




if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task1")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task1", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task1", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task1", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)



