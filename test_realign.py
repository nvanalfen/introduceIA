from halotools.empirical_models import DimrothWatson
from halotools.utils.rotations3d import rotation_matrices_from_vectors
from halotools.utils.mcrotations import random_unit_vectors_3d
from halotools.utils.vector_utilities import rotate_vector_collection, normalized_vectors
from halotools.utils import angles_between_list_of_vectors, elementwise_dot
from scipy.spatial.transform import Rotation as R
import numpy as np
from realign import align_vectors, perpendicular_on_plane, partial_rotation_matrices_from_vectors

#############################################################################################################################
##### TEST HELPER FUNCTIONS ################################################################################################
#############################################################################################################################

def test_perpendicular_on_plane():
    m = 1_000_000
    vecs = random_unit_vectors_3d(m)
    refs = random_unit_vectors_3d(m)
    perp_refs = perpendicular_on_plane(refs, vecs)

    assert np.allclose(elementwise_dot(perp_refs, refs), 0, rtol=1e-5)                          # Check perpendicularity
    assert np.allclose( elementwise_dot(perp_refs, np.cross(refs, vecs)), 0, rtol=1e-5 )        # Check in-plane

def test_partial_rotation_matrices_from_vectors():
    # Double check with large lists of vectors
    N = 1_000_000
    vecA_large, vecB_large = random_unit_vectors_3d(N), random_unit_vectors_3d(N)
    vecB_large *= np.sign(elementwise_dot(vecA_large, vecB_large))[:, np.newaxis]
    mat_large = rotation_matrices_from_vectors(vecA_large, vecB_large)

    full_mat_large = partial_rotation_matrices_from_vectors(vecA_large, vecB_large, 1.0)
    quarter_mat_large = partial_rotation_matrices_from_vectors(vecA_large, vecB_large, 0.25)

    # First, check that the large list of vectors is rotated correctly using the normal rotation matrix
    vecC_large = rotate_vector_collection(mat_large, vecA_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert np.allclose(dots, 1, rtol=1e-5)

    # Check that the partial function can recreate the full rotation matrix
    vecC_large = rotate_vector_collection(full_mat_large, vecA_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert np.allclose(dots, 1, rtol=1e-5)

    # Now, check that the large list of vectors is rotated correctly using the partial rotation matrix
    vecC_large = rotate_vector_collection(quarter_mat_large, vecA_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert not np.allclose(dots, 1, rtol=1e-5)
    vecC_large = rotate_vector_collection(quarter_mat_large, vecC_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert not np.allclose(dots, 1, rtol=1e-5)
    vecC_large = rotate_vector_collection(quarter_mat_large, vecC_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert not np.allclose(dots, 1, rtol=1e-5)
    vecC_large = rotate_vector_collection(quarter_mat_large, vecC_large)
    dots = elementwise_dot(vecC_large, vecB_large)
    assert np.allclose(dots, 1, rtol=1e-5)

#############################################################################################################################
##### TEST FULL ALIGNMENT ##################################################################################################
#############################################################################################################################

def test_float_align_parallel():
    """
    Test the align_vectors function with float input for parallel alignment.
    """
    m = 1_000_000
    ref_x = np.array([ np.ones(m), np.zeros(m), np.zeros(m) ]).T
    vecs = normalized_vectors( np.array([ np.random.uniform(0, 1, size=m), np.random.uniform(0, 1, size=m), np.zeros(m) ]).T )
    initial_angles = angles_between_list_of_vectors(vecs, ref_x)

    # Rotate towards the reference vectors
    rot_vecs = align_vectors(vecs, ref_x, 1)                        # Perfect Rotation
    assert np.allclose(rot_vecs, ref_x, rtol=1e-5)
    rot_vecs = align_vectors(vecs, ref_x, 0.5)                      # Half Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_x)
    assert np.allclose(rot_angles, initial_angles / 2, rtol=1e-5)
    rot_vecs = align_vectors(vecs, ref_x, 0)                        # No Rotation
    assert np.allclose(rot_vecs, vecs, rtol=1e-5)
    rand_fracs = np.random.rand(m)
    rot_vecs = align_vectors(vecs, ref_x, rand_fracs)               # Random Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_x)
    assert np.allclose(rot_angles, initial_angles * (1-rand_fracs), rtol=1e-3)

def test_float_align_perpendicular():
    """
    Test the align_vectors function with float input for perpendicular alignment.
    """
    m = 1_000_000
    ref_x = np.array([ np.ones(m), np.zeros(m), np.zeros(m) ]).T
    ref_y = np.array([ np.zeros(m), np.ones(m), np.zeros(m) ]).T
    vecs = normalized_vectors( np.array([ np.random.uniform(0, 1, size=m), np.random.uniform(0, 1, size=m), np.zeros(m) ]).T )
    initial_angles = angles_between_list_of_vectors(vecs, ref_x)
    complement_angles = np.pi/2 - initial_angles

    # Rotate towards the vectors perpendicular to reference
    rot_vecs = align_vectors(vecs, ref_x, -1)                        # Perfect Rotation
    assert np.allclose(rot_vecs, ref_y, rtol=1e-5)
    rot_vecs = align_vectors(vecs, ref_x, -0.5)                      # Half Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_x)
    assert np.allclose(rot_angles, initial_angles + (complement_angles / 2 ), rtol=1e-5)
    rot_vecs = align_vectors(vecs, ref_x, 0)                        # No Rotation
    assert np.allclose(rot_vecs, vecs, rtol=1e-5)
    rand_fracs = np.random.rand(m)
    rot_vecs = align_vectors(vecs, ref_x, -rand_fracs)         # Random Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_x)
    assert np.allclose(rot_angles, initial_angles + ( complement_angles * rand_fracs ), rtol=1e-3)

def test_array_align_parallel():
    # Rotate towards parallel
    # Pass in random array of fractions (all positive)
    m = 1_000_000
    rand_fracs = np.random.uniform(0, 1, size=m)
    ref_vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    close_vecs = np.sign(elementwise_dot(vecs, ref_vecs))[:, np.newaxis] * vecs                     # For looking at the smaller angle
    initial_angles = angles_between_list_of_vectors(close_vecs, ref_vecs)

    rot_vecs = align_vectors(vecs, ref_vecs, rand_fracs)                            # Random Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_vecs)                 # Check the angles
    expected_angles = initial_angles * (1 - np.abs(rand_fracs))                     # Reduce angle for those rotating towards parallel
    expected_angles = np.minimum(expected_angles, np.pi-expected_angles)            # Ensure angles are less than 90 degrees
    rot_angles = np.minimum(rot_angles, np.pi-rot_angles)                           # Ensure angles are less than 90 degrees

    assert np.allclose(rot_angles, expected_angles, rtol=1e-3)                      # Check the angles
    assert(rot_angles <= np.pi/2).all()                                             # Ensure angles are less than 90 degrees
    assert (rot_angles <= initial_angles).all()                                     # Ensure angles are less than the initial angles
    assert (rot_angles >= 0).all()                                                  # Ensure angles are positive

def test_array_align_perpendicular():
    # Rotate towards perpendicular
    # Pass in random array of fractions (all negative)
    m = 1_000_000
    rand_fracs = np.random.uniform(-1, 0, size=m)
    ref_vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    close_vecs = np.sign(elementwise_dot(vecs, ref_vecs))[:, np.newaxis] * vecs                     # For looking at the smaller angle
    initial_angles = angles_between_list_of_vectors(close_vecs, ref_vecs)
    complement_angles = np.pi/2 - initial_angles

    rot_vecs = align_vectors(vecs, ref_vecs, rand_fracs)                            # Random Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_vecs)                 # Check the angles
    delta_theta = complement_angles * np.abs(rand_fracs)                            # Increase angle for those rotating towards perpendicular
    expected_angles = initial_angles + delta_theta                                  # Reduce angle for those rotating towards parallel
    expected_angles = np.minimum(expected_angles, np.pi-expected_angles)            # Ensure angles are less than 90 degrees
    rot_angles = np.minimum(rot_angles, np.pi-rot_angles)                           # Ensure angles are less than 90 degrees

    assert np.allclose(rot_angles, expected_angles, rtol=1e-3)                      # Check the angles
    assert(rot_angles >= initial_angles).all()                                      # Ensure angles are greater than the initial angles
    assert (rot_angles <= np.pi/2).all()                                            # Ensure angles are less than 90 degrees
    assert (rot_angles >= 0).all()                                                  # Ensure angles are positive

def test_array_align_mixed():
    # Rotate either towards parallel or towards perpendicular
    # Pass in random array of fractions
    m = 1_000_000
    rand_fracs = np.random.uniform(-1, 1, size=m)
    ref_vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    vecs = normalized_vectors( np.array( [ np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m), np.random.uniform(-1, 1, size=m) ] ).T )
    close_vecs = np.sign(elementwise_dot(vecs, ref_vecs))[:, np.newaxis] * vecs                     # For looking at the smaller angle
    initial_angles = angles_between_list_of_vectors(close_vecs, ref_vecs)
    complement_angles = np.pi/2 - initial_angles
    perp_mask = rand_fracs < 0

    rot_vecs = align_vectors(vecs, ref_vecs, rand_fracs)                                            # Random Rotation
    rot_angles = angles_between_list_of_vectors(rot_vecs, ref_vecs)                                 # Check the angles
    # Handle the parallel cases
    expected_angles = initial_angles * (1 - np.abs(rand_fracs))                                     # Reduce angle for those rotating towards parallel
    # Handle the perpendicular cases
    delta_theta = complement_angles[perp_mask] * np.abs(rand_fracs[perp_mask])                      # Increase angle for those rotating towards perpendicular
    expected_angles[perp_mask] = initial_angles[perp_mask] + delta_theta                            # Reduce angle for those rotating towards parallel
    expected_angles = np.minimum(expected_angles, np.pi-expected_angles)                            # Ensure angles are less than 90 degrees
    rot_angles = np.minimum(rot_angles, np.pi-rot_angles)                                           # Ensure angles are less than 90 degrees

    assert(np.allclose(rot_angles[~perp_mask], expected_angles[~perp_mask], rtol=1e-3))             # Check the angles for parallel
    assert(np.allclose(rot_angles[perp_mask], expected_angles[perp_mask], rtol=1e-3))               # Check the angles for perpendicular
    assert (rot_angles[~perp_mask] <= initial_angles[~perp_mask]).all()                             # Make sure all + fractions rotate towards
    assert (rot_angles[perp_mask] >= initial_angles[perp_mask]).all()                               # Make sure all - fractions rotate away
    assert (rot_angles <= np.pi/2).all()                                                            # Ensure angles are less than 90 degrees
    assert (rot_angles >= 0).all()                                                                  # Ensure angles are positive

if __name__ == "__main__":
    print("Testing...")

    # Test Helper Functions
    print("Helper Function Tests...")
    test_perpendicular_on_plane()
    test_partial_rotation_matrices_from_vectors()

    print("Full Alignment Tests...")
    # Test alignment with single float fraction
    test_float_align_parallel()
    test_float_align_perpendicular()

    # Test alignment with array of fractions
    test_array_align_parallel()
    test_array_align_perpendicular()
    test_array_align_mixed()

    print("Tests Passed!")