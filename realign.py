from halotools.empirical_models import DimrothWatson
from halotools.utils.rotations3d import rotation_matrices_from_angles
from halotools.utils.vector_utilities import rotate_vector_collection
from halotools.utils import elementwise_dot
from scipy.spatial.transform import Rotation as R

import numpy as np

def quat_rotation_matrix_from_vectors(vecA, vecB):
    """
    Compute the quaternion that rotates vecA to vecB.
    """
    vecA = vecA / np.linalg.norm(vecA, axis=1)[:, np.newaxis]
    vecB = vecB / np.linalg.norm(vecB, axis=1)[:, np.newaxis]
    half = (vecA + vecB) / np.linalg.norm(vecA + vecB, axis=1)[:, np.newaxis]
    w = np.sum(vecA * half, axis=1)
    cross_product = np.cross(vecA, half)
    x, y, z = cross_product.T
    quats = np.column_stack((x, y, z, w))
    return R.from_quat(quats)

def slerp_quaternion(quatA, quatB, fraction):
    """
    Perform spherical linear interpolation (Slerp) between two quaternions.
    """
    fraction = np.ones(len(quatA)) * fraction
    dot_product = np.sum(quatA * quatB, axis=1)

    quatB[dot_product < 0] = -quatB[dot_product < 0]
    dot_product = np.abs(dot_product)

    close_mask = dot_product > 0.9995
    far_mask = ~close_mask

    interpolated_quat = np.empty_like(quatA, dtype=np.float64)
    theta_0 = np.arccos(dot_product)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * fraction
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    interpolated_quat[close_mask] = quatA[close_mask] + fraction[close_mask][:, np.newaxis] * (quatB[close_mask] - quatA[close_mask])
    interpolated_quat[far_mask] = (s0[far_mask][:, np.newaxis] * quatA[far_mask]) + (s1[far_mask][:, np.newaxis] * quatB[far_mask])

    interpolated_quat /= np.linalg.norm(interpolated_quat, axis=1)[:, np.newaxis]
    return interpolated_quat

def partial_rotation_matrices_from_vectors(vecA, vecB, fraction):
    """
    Compute the rotation matrices for partial rotations from vecA to vecB.
    """
    rot_full = quat_rotation_matrix_from_vectors(vecA, vecB)
    quat_full = rot_full.as_quat()
    quat_id = np.array([[0, 0, 0, 1]] * len(vecA))  # Identity quaternions

    quat_partial = slerp_quaternion(quat_id, quat_full, fraction)

    rot_partial = R.from_quat(quat_partial)
    return rot_partial.as_matrix()

def perpendicular_on_plane(vecA, vecB):
    """
    Find the vector perpendicular to vecA on the plane defined by vecA and vecB.
    """
    signs = np.sign(elementwise_dot(vecA, vecB))
    vecC = signs[:,np.newaxis] * vecB
    directions = np.cross(vecA, vecC)
    mats = rotation_matrices_from_angles(np.pi/2 * np.ones(len(vecA)), directions)
    return rotate_vector_collection(mats, vecA)

def align_vectors(vecA, vecB, fraction):
    """
    Rotates vecA some portion of the way towards vecB.
    Negative fraction will align towards the perpendicular vector to vecB.
    Positive fraction will align towards the nearest end of vecB (i.e. vecB or -vecB).
    
    inputs:
    -------
    vecA : array_like, Nx3
        The vectors to be rotated.
    vecB : array_like, Nx3
        The reference direction
    fraction : float or array_like (Nx3)
        The fraction of the way towards vecB to rotate vecA.
        If float, the same fraction will be used for all vectors.
        If array_like, the fraction for each vector.
    """
    fraction = np.ones(len(vecA)) * fraction
    ref_vecs = np.array(vecB)

    # Compute the rotation matrices
    perpendicular_mask = fraction < 0
    if perpendicular_mask.any():
        # Not that vecB is the first vector passed in
        # This is because we want the vector perpendicular to vecB on the plane defined by vecB and vecA
        ref_vecs[perpendicular_mask] = perpendicular_on_plane(vecB[perpendicular_mask], vecA[perpendicular_mask])

    # Ensure we use just the nearest side of the reference vector
    signs = np.sign(elementwise_dot(vecA, ref_vecs))
    ref_vecs = signs[:,np.newaxis] * ref_vecs
    mats = partial_rotation_matrices_from_vectors(vecA, ref_vecs, abs(fraction))
    
    return rotate_vector_collection(mats, vecA)