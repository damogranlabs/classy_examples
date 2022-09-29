import unittest

import numpy as np

from classy_blocks.classes.operations import Face, Loft, Extrude, Revolve
from classy_blocks.util import functions as f

class OperationTests(unittest.TestCase):
    def setUp(self):
        self.bottom_face = Face(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
            [
                [0.5, -0.25, 0], None, None, None
            ]
        )
        self.top_face = Face(
            [
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            [
                [0.5, 0.25, 1], None, None, None
            ]
        )

    def test_extrude_create(self):
        extrude = Extrude(self.bottom_face, [0, 0, 1])
        self.assertEqual(len(extrude.block.edges), 2)

    def test_loft_create(self):
        # add 2 curved side edges
        loft = Loft(
            self.bottom_face,
            self.top_face,
            [
                [0.1, 0.1, 0.5],
                [1.1, 0.1, 0.5],
                None,
                None
            ]
        )

        # two curved edges were specified
        self.assertEqual(len(loft.block.edges), 4)

    def test_revolve_create(self):
        revolve = Revolve(self.bottom_face, np.pi/3, [0, 1, 0], [2, 0, 0])
        # 4 curved edges are added
        self.assertEqual(len(revolve.block.edges), 6)

class TransformationTests(unittest.TestCase):
    def setUp(self):
        face_points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]

        face_edges = [
            [0.5, -0.25, 0], None, None, None
        ]

        self.face = Face(face_points, face_edges)
        self.vector = np.array([0, 0, 1])
        self.extrude = Extrude(
            base=self.face,
            extrude_vector=self.vector
        )

    def test_translate(self):
        translate_vector = [0, 0, 1]

        original_op = self.extrude
        translated_op = self.extrude.copy().translate(translate_vector)

        np.testing.assert_almost_equal(
            original_op.bottom_face.points + translate_vector,
            translated_op.bottom_face.points
        )

        np.testing.assert_almost_equal(
            original_op.edges[0].points + translate_vector,
            translated_op.edges[0].points
        )

    def test_rotate(self):
        axis = [0, 1, 0]
        origin = [0, 0, 0]
        angle = np.pi/2

        original_op = self.extrude
        rotated_op = self.extrude.copy().rotate(axis, angle, origin)

        def extrude_direction(op):
            return op.top_face.points[0] - op.bottom_face.points[0]

        np.testing.assert_almost_equal(
            f.angle_between(extrude_direction(original_op),
            extrude_direction(rotated_op)),
            angle
        )

if __name__ == '__main__':
    unittest.main()