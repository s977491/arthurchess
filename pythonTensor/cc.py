'''
A board is a NxN numpy array.
A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).
A PlayerMove is a (Color, Move) tuple

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
'''
from collections import namedtuple
import copy
import itertools

import numpy as np

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
BLACK, EMPTY, RED = range(-1, 2)

class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])): pass

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

class IllegalMove(Exception): pass

# these are initialized by set_board_size
Nx = None
Ny = None
ALL_COORDS = []
NEIGHBORS = {}
DIAGONALS = {}

def set_board_size():
    '''
    Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.
    Also, never do "from go import N, W, ALL_COORDS, EMPTY_BOARD".
    '''
    global Nx, Ny, ALL_COORDS, NEIGHBORS, DIAGONALS
    Ny = 10
    Nx = 9
    ALL_COORDS = [(i, j) for i in range(Ny) for j in range(Nx)]
    def check_bounds(c):
        return c[0] % Ny == c[0] and c[1] % Nx == c[1]

    NEIGHBORS = {(x, y): list(filter(check_bounds, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
    DIAGONALS = {(x, y): list(filter(check_bounds, [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}


class Position():
    def __init__(self, board, moveFrom, moveTo, win, step, lastMoveFrom=None, lastMoveTo=None):
        '''
        board: a numpy array
        n: an int representing moves played so far
        komi: a float, representing points given to the second player.
        caps: a (int, int) tuple of captures for B, W.
        lib_tracker: a LibertyTracker object
        ko: a Move
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        to_play: BLACK or WHITE
        '''
        self.board = board
        self.moveFrom = moveFrom
        self.moveTo = moveTo
        self.win = win
        self.step = step
        self.lastMoveFrom = lastMoveFrom
        self.lastMoveTo = lastMoveTo

#    def __deepcopy__(self, memodict={}):
#        new_board = np.copy(self.board)
#        return Position(new_board, self.n, self.komi, self.caps, new_lib_tracker, self.ko, self.recent, self.to_play)

    # def __str__(self):
    #     pretty_print_map = {
    #         BLACK: '\x1b[0;31;47mO',
    #         EMPTY: '\x1b[0;31;43m.',
    #         RED: '\x1b[0;31;40mX',
    #
    #     }
    #     board = np.copy(self.board)
    #     captures = self.caps
    #     if self.ko is not None:
    #         place_stones(board, KO, [self.ko])
    #     raw_board_contents = []
    #     for i in range(N):
    #         row = []
    #         for j in range(N):
    #             appended = '<' if (self.recent and (i, j) == self.recent[-1].move) else ' '
    #             row.append(pretty_print_map[board[i,j]] + appended)
    #             row.append('\x1b[0m')
    #         raw_board_contents.append(''.join(row))
    #
    #     row_labels = ['%2d ' % i for i in range(N, 0, -1)]
    #     annotated_board_contents = [''.join(r) for r in zip(row_labels, raw_board_contents, row_labels)]
    #     header_footer_rows = ['   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
    #     annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
    #     details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n, *captures)
    #     return annotated_board + details


set_board_size()