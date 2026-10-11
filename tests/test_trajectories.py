import numpy as np

from g1_playback.trajectories import minimum_jerk, smooth_segment


def test_minimum_jerk_endpoints():
    y = minimum_jerk(np.array([0.0, 1.0]))
    np.testing.assert_allclose(y, [0.0, 1.0])


def test_smooth_segment_endpoints():
    p0 = np.array([0.0, 1.0, 2.0])
    p1 = np.array([3.0, 4.0, 5.0])
    traj = smooth_segment(p0, p1, 20)
    np.testing.assert_allclose(traj[0], p0)
    np.testing.assert_allclose(traj[-1], p1)
