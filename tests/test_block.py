import unittest

import numpy as np

from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
from classy_blocks.util import constants

from tests.fixtures import FixturedTestCase

class TestBlock(FixturedTestCase):
    ###
    ### Block tests
    ###
    def test_block_definition(self):
        """ the correct output for blockMeshDict """
        self.mesh.prepare_data()

        self.assertEqual(
            str(self.block_0),
            "hex  ( 0 1 2 3 4 5 6 7 )  (6 6 7)  simpleGrading (1 1 1) // 0 Test"
        )

    def test_any_grading_defined(self):
        self.mesh.prepare_data()
        
        self.block_1.grading = [None, None, None]
        self.assertFalse(self.block_1.is_any_grading_defined)

        self.block_1.grading = [1, None, None]
        self.assertTrue(self.block_1.is_any_grading_defined)

    def test_any_count_defined(self):
        self.mesh.prepare_data()

        self.block_1.n_cells = [None, None, None]
        self.assertFalse(self.block_1.is_any_count_defined)
        
        self.block_1.n_cells = [1, None, None]
        self.assertTrue(self.block_1.is_any_count_defined)

    def test_block_grading_undefined(self):
        """ where grading of the block is None, use 1 """
        self.mesh.prepare_data()
        self.block_0.grading = [None, None, None]

        self.assertEqual(
            str(self.block_0),
            "hex  ( 0 1 2 3 4 5 6 7 )  (6 6 7)  simpleGrading (1 1 1) // 0 Test"
        )

    def test_is_count_not_defined(self):
        """ block.is_count_defined must be True if any of block.n_cells is None """
        self.mesh.prepare_data()

        self.block_1.n_cells = [None, None, None]
        self.assertFalse(self.block_1.is_count_defined)

        self.block_1.n_cells = [0, 0, 0]
        self.assertFalse(self.block_1.is_count_defined)

        self.block_1.n_cells = [0, 1, 10]
        self.assertFalse(self.block_1.is_count_defined)

    def test_is_count_defined(self):
        """ block.is_count_defined must be True if any of block.n_cells is None """
        self.mesh.prepare_data()

        self.block_1.n_cells = [1, 10, 2]
        self.assertTrue(self.block_1.is_count_defined)
    
    def test_face_format(self):
        """ the correct face format for blockMeshDict """
        self.mesh.prepare_data()

        self.assertEqual(
            self.block_0.format_face('bottom'),
            "(0 1 2 3)"
        )
    
    def test_patches(self):
        """ patch naming/positions """
        self.mesh.prepare_data()

        self.assertListEqual(self.block_0.patches['inlet'], ['left'])
        self.assertListEqual(self.block_2.patches['outlet'], ['back'])

        self.assertListEqual(
            self.block_0.patches['walls'], ['bottom', 'top', 'front', 'back'])

        self.assertListEqual(self.block_1.patches['walls'], ['bottom', 'top', 'right', 'front'])

        self.assertListEqual(self.block_2.patches['outlet'], ['back'])
        self.assertListEqual(self.block_2.patches['walls'], ['bottom', 'top', 'left', 'right'])

    def test_faces(self):
        """ definitions of faces around the block """
        self.mesh.prepare_data()

        self.assertListEqual(
            self.block_0.get_faces('walls'),
            ['(0 1 2 3)', '(4 5 6 7)', '(4 5 1 0)', '(7 6 2 3)']
        )

    def test_straight_block_size(self):
        """ length of a straight block edge """
        self.mesh.prepare_data()

        self.assertEqual(self.block_1.get_size(2), 1)
    
    def test_arc_block_size(self):
        """ length of a curved block edge (two segments) """
        self.mesh.prepare_data()

        self.assertAlmostEqual(
            self.block_0.get_size(0), 1.0295084971874737
        )

    def test_spline_block_size(self):
        """ length of a spline block edge (three or more segments) """
        self.mesh.prepare_data()

        self.assertAlmostEqual(self.block_0.get_size(1), 1.0121046080181824)

    def test_cell_size(self):
        """ grade_to_size must calculate the correct grading to match given cell size """
        axis = 0
        test_cell_size = 0.005
        n = self.block_0.n_cells[axis]

        # set the desired cell size
        self.block_0.grade_to_size(axis, test_cell_size)
        self.mesh.prepare_data() # runs all deferred functions

        # get block grading
        g = self.block_0.grading[axis]

        # check that the sum of all elements, graded, is block size
        cell_sizes = [test_cell_size]
        block_size = 0

        for _ in range(n):
            s = cell_sizes[-1]*(g**(1/n))
            cell_sizes.append(s)
            block_size += s

        self.assertAlmostEqual(block_size, self.block_0.get_size(axis))

        # also check that ratio of first to last cell size is what's calculated
        self.assertAlmostEqual(cell_sizes[-1]/cell_sizes[0], g, delta=constants.tol)

    def test_axis_from_pair(self):
        """ return the correct pairs of points along each axis """
        self.mesh.prepare_data()

        # this only works for the first block, where
        # index in block.vertices[] and vertex.mesh_index coincide;
        # later these numbers differ
        pairs = [
            [[0, 1], [3, 2], [4, 5], [7, 6]],
            [[0, 3], [1, 2], [5, 6], [4, 7]],
            [[0, 4], [1, 5], [2, 6], [3, 7]],
        ]

        for i, pair in enumerate(pairs):
            for j in range(4):
                v1 = self.block_1.vertices[pair[j][0]].mesh_index
                v2 = self.block_1.vertices[pair[j][1]].mesh_index

                self.assertEqual(
                    self.block_1.get_axis_from_pair((v1, v2)), i)

    def test_axis_vertex_pairs(self):
        """ pairs of vertices for wedges """ 
        # create a three-sided pyramid;
        # get_axis_vertex_pairs should return less pairs for triangular faces
        block_points = [
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [0.5, 1, 0], # 2
            [0.5, 1, 0], # 3

            [0, 0, 1], # 4
            [1, 0, 1], # 5
            [0.5, 1, 1], # 6
            [0.5, 1, 1], # 7
        ]

        block = Block.create_from_points(block_points)
        block.n_cells = [10, 10, 10]

        mesh = Mesh()
        mesh.add_block(block)
        mesh.prepare_data()

        self.assertEqual(len(block.get_axis_vertex_pairs(0)), 2)
        self.assertEqual(len(block.get_axis_vertex_pairs(1)), 4)
        self.assertEqual(len(block.get_axis_vertex_pairs(2)), 3)

    def test_copy_grading(self):
        """ grading propagation """
        self.block_0.grade_to_size(2, 0.01)

        mesh = Mesh()
        mesh.add_block(self.block_0)
        mesh.add_block(self.block_1)
        mesh.add_block(self.block_2)
        mesh.prepare_data()

        self.assertIsNotNone(self.block_1.grading[2])
        self.assertIsNotNone(self.block_2.grading[2])

if __name__ == '__main__':
    unittest.main()