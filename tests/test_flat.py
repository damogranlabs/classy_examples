import unittest

import numpy as np

from classy_blocks.classes.flat.face import Face
from classy_blocks.util import functions as f

class FaceTests(unittest.TestCase):
    def setUp(self):
        self.points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]   

    def test_points(self):
        # provide less than 4 points
        with self.assertRaises(Exception):
            Face(self.points[:3])
    
    def test_edges(self):
        with self.assertRaises(Exception):
            Face(
                self.points,
                [None, None, None]
            )

    def test_coplanar_points_fail(self):
        with self.assertRaises(Exception):
            self.points[-1][-1] = 0.1
            Face(self.points, check_coplanar=True)

    def test_coplanar_points_success(self):
        Face(self.points, check_coplanar=True)

    def test_translate(self):
        face_edges = [
            [0.5, -0.25, 0], # arc edge
            [[1.1, 0.25, 0], [1.2, 0.5, 0], [1.1, 0.75, 0]], # spline edge
            None,
            None
        ]

        translate_vector = np.random.rand(3)

        original_face = Face(self.points, face_edges)
        translated_face = original_face.copy().translate(translate_vector)
        
        # check points
        for i in range(4):
            p1 = original_face.points[i]
            p2 = translated_face.points[i]

            np.testing.assert_almost_equal(p1, p2 - translate_vector)

        # check arc edge
        np.testing.assert_almost_equal(
            translated_face.edges[0].points - translate_vector,
            original_face.edges[0].points
        )

        # check spline edge
        np.testing.assert_array_almost_equal(
            translated_face.edges[1].points - translate_vector,
            original_face.edges[1].points
        )

    def test_points(self):
        np.testing.assert_array_almost_equal(
            self.points, Face(self.points).points
        )

    def test_get_edge_data(self):
        edge_point = [0.5, 0.1, 0]
        edge_data = [None, None, edge_point, None]

        face = Face(self.points, edge_data)

        self.assertListEqual(edge_data, face.get_edge_data())

    def test_copy(self):
        original = Face(self.points)
        copied = original.copy()

        for v in copied.vertices:
            self.assertIsNone(v.index)

    def test_rotate(self):
        # only test that the Face.rotate function works properly;
        # other machinery (translate, transform_points, transform_edges) are tested in 
        # test_translate above
        origin = np.random.rand(3)
        angle = np.pi/3
        axis = np.array([1, 1, 1])

        original_face = Face(self.points)
        rotated_face = original_face.copy().rotate(axis, angle, origin)
        
        for i in range(4):
            original_point = original_face.points[i]
            rotated_point = rotated_face.points[i]

            np.testing.assert_almost_equal(
                rotated_point,
                f.arbitrary_rotation(original_point, axis, angle, origin)
            )

    def test_scale_default_origin(self):
        original_face = Face(self.points)
        scaled_face = original_face.copy().scale(2)

        scaled_points = [
            [-0.5, -0.5,  0],
            [ 1.5, -0.5,  0],
            [ 1.5,  1.5,  0],
            [-0.5,  1.5,  0]
        ]

        np.testing.assert_array_almost_equal(scaled_face.points, scaled_points)

    def test_scale_custom_origin(self):
        original_face = Face(self.points)
        scaled_face = original_face.copy().scale(2, [0, 0, 0])

        scaled_points = np.array(self.points)*2

        np.testing.assert_array_almost_equal(scaled_face.points, scaled_points)

    def test_scale_edges(self):
        original_edges = [[0.5, -0.25, 0], None, None, None]
        original_face = Face(self.points, original_edges)

        scaled_face = original_face.copy().scale(2, origin=[0, 0, 0])

        self.assertListEqual(scaled_face.edges[0].tolist(), [1, -0.5, 0])

    def test_invert(self):
        face = Face(self.points)
        face.invert()

        np.testing.assert_array_almost_equal(face.points, np.flip(self.points, axis=0))

class CircleTests(unittest.TestCase):
    # TODO
    pass

class AnnulusTests(unittest.TestCase):
    # TODO
    pass