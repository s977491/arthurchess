#!/usr/bin/python3

import numpy as np
import ctypes
# from numpy.ctypes import c_ndarray

BOARD_START = [
    "rh a ke  ",
    "    ar   ",
    " c  e h c",
    "p p p p p",
    "         ",
    "         ",
    "P P P P P",
    " CH E  C ",
    "   RAH   ",
    "  EK A  R"

]

def get_start_board():
    Matrix = np.zeros([10, 9], dtype=np.int8)
    for y,pieceLine in enumerate(BOARD_START):
        for x, piece in enumerate(list(pieceLine)):
            Matrix[y,x] = ord(piece)
    return Matrix

m = get_start_board()
import archess
archess.getMaxEatMove(m.tolist())
