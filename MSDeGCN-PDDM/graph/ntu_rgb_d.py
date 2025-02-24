import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 17
self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(17, 15), (15, 0), (0, 16), (16, 18), (0, 1),
#                     (1, 5), (5, 6), (6, 7), (1, 2), (2, 3), (3, 4),
#             (1, 8), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21), (8, 9), (9, 10), (10, 11),
#             (11, 22), (22, 23), (11, 24)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_ori_index = [(1, 2), (2, 3), (3, 4),  (5, 1), (6, 5), (7, 6),
                    (8, 1), (9, 8), (10, 9), (11, 10), (12, 9), (13, 12),
                    (14, 13), (15, 9), (16, 15), (17, 6)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


