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

BOARD_START = [
    "rheakaehr",
    "         ",
    " c     c ",
    "p p p p p",
    "         ",
    "         ",
    "P P P P P",
    " C     C ",
    "         ",
    "RHEAKAEHR"
]

def get_start_board():
    Matrix = np.zeros([Ny, Nx], dtype=np.int8)
    for y,pieceLine in enumerate(BOARD_START):
        for x, piece in enumerate(list(pieceLine)):
            Matrix[y,x] = ord(piece)

    return Position(Matrix, None, None, 1, 0 )



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

class GameMetadata(namedtuple("GameMetadata", "result step")):
    pass

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
        self.v = 0.0

    def clone(self):
        return Position(np.copy(self.board), self.moveFrom, self.moveTo,self.win, self.step, self.lastMoveFrom, self.lastMoveTo )
    def move(self, moveFrom, moveTo):
        self.lastMoveFrom = self.moveFrom
        self.lastMoveTo = self.moveTo
        self.moveTo = moveTo
        self.moveFrom = moveFrom

        ret =self.clone() # snapshot for leanring
        self.board[moveTo] = self.board[moveFrom]
        self.board[moveFrom] = ord(' ')
        return ret

    def validMov(self, m, allowSame=False):
        for i in m:
            if i < 0:
                return False
        if m[0] >= Ny or m[2] >=Ny or m[1] >= Nx or m[3] >= Nx:
            return False

        if allowSame:
            return True
        if self.board[m[2], m[3]] < ord('a'):
            return True
        return False

    def pcChecker(self, mov, ob, movelist):
        finish = False
        if not self.validMov(mov, True):
            finish = True
        elif ob == 0:
            if self.board[mov[2], mov[3]] == ord(' '):
                movelist.append(mov)
            else:
                ob += 1
        else:
            if self.board[mov[2], mov[3]] != ord(' '):
                if  self.board[mov[2], mov[3]] <= ord('Z'):
                    movelist.append(mov)
                finish = True

        return finish, ob

    def possibleMoves(self):
        ret = []
        hasKing = False
        shape = self.board.shape
        for y in range(shape[0]):
            for x in range(shape[1]):
                if self.board[y, x] >= ord('a'):
                    src = self.board[y, x]
                    if src == ord('k'):
                        hasKing = True
                        for i in range(2):
                            for sign in range(-1, 2, 2):
                                j = 1 -i
                                toY = y + i * sign
                                toX = x + j * sign
                                m = (y, x, toY, toX)
                                if self.validMov(m) and toY <=2 and toX >=3 and toX <=5:
                                    ret.append(m)
                    elif src == ord('r'):
                        toY = y
                        for toX in range (x -1, -1, -1):
                            m = (y, x, toY, toX)
                            if not self.validMov(m):
                                break
                            ret.append(m)
                            if self.board[toY, toX] != ord(' '):
                                break
                        for toX in range (x +1, Nx):
                            m = (y, x, toY, toX)
                            if not self.validMov(m):
                                break
                            ret.append(m)
                            if self.board[toY, toX] != ord(' '):
                                break
                        toX = x
                        for toY in range (y -1, -1, -1):
                            m = (y, x, toY, toX)
                            if not self.validMov(m):
                                break
                            ret.append(m)
                            if self.board[toY, toX] != ord(' '):
                                break
                        for toY in range (y +1, Ny):
                            m = (y, x, toY, toX)
                            if not self.validMov(m):
                                break
                            ret.append(m)
                            if self.board[toY, toX] != ord(' '):
                                break
                    elif src == ord('h'):
                        for i in range(-1,3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                m = (y, x, toY, toX)
                                if self.validMov(m) and self.board[toY - i, x] == ord(' '):
                                    ret.append(m)
                                toY = y + 1 * i
                                toX = x + 2 * j
                                m = (y, x, toY, toX)
                                if self.validMov(m) and self.board[y, toX - j] == ord(' '):
                                    ret.append(m)
                    elif src == ord('c'):
                        toY = y
                        obstacle = 0
                        for toX in range(x - 1, -1, -1):
                            m = (y, x, toY, toX)
                            finish, obstacle = self.pcChecker(m, obstacle, ret)
                            if finish:
                                break
                        obstacle = 0
                        for toX in range(x + 1, Nx):
                            m = (y, x, toY, toX)
                            finish, obstacle = self.pcChecker(m, obstacle, ret)
                            if finish:
                                break
                        toX = x
                        obstacle = 0
                        for toY in range(y - 1, -1, -1):
                            m = (y, x, toY, toX)
                            finish, obstacle = self.pcChecker(m, obstacle, ret)
                            if finish:
                                break
                        obstacle = 0
                        for toY in range(y + 1, Ny):
                            m = (y, x, toY, toX)
                            finish, obstacle = self.pcChecker(m, obstacle, ret)
                            if finish:
                                break
                    elif src == ord('a'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i
                            m = (y, x, toY, toX)
                            if self.validMov(m) and toY <= 2 and toX >= 3 and toX <= 5:
                                ret.append(m)

                            toY = y + i
                            toX = x - i
                            m = (y, x, toY, toX)
                            if self.validMov(m) and toY <= 2 and toX >= 3 and toX <= 5:
                                ret.append(m)

                    elif src == ord('e'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i
                            m = (y, x, toY, toX)
                            if self.validMov(m) and toY <= 4 and self.board[y+i//2, x+i//2] == ord(' '):
                                ret.append(m)
                            toY = y + i
                            toX = x - i
                            m = (y, x, toY, toX)
                            if self.validMov(m) and toY <= 4 and self.board[y+i//2, x-i//2] == ord(' '):
                                ret.append(m)
                    elif src == ord('p'):
                        toY = y + 1
                        toX = x
                        m = (y, x, toY, toX)
                        if self.validMov(m):
                            ret.append(m)
                        if y > 4:
                            toY = y
                            toX = x + 1
                            m = (y, x, toY, toX)
                            if self.validMov(m):
                                ret.append(m)
                            toX = x - 1
                            m = (y, x, toY, toX)
                            if self.validMov(m):
                                ret.append(m)
        if hasKing:
            return ret
        return []

    def flip(self):
        activePlayerPiece = self.board >= ord('a')
        passivePlayerPiece = np.logical_and(self.board < ord('a'), self.board != ord(' '));
        caseDiff =  (ord('a')- ord('A'))
        self.board[activePlayerPiece] = self.board[activePlayerPiece] - caseDiff
        self.board[passivePlayerPiece] = self.board[passivePlayerPiece] + caseDiff
        self.board = np.flipud(np.fliplr(self.board))
        if self.lastMoveFrom:
            self.lastMoveFrom = (Ny-self.lastMoveFrom[0] -1, Nx-self.lastMoveFrom[1] -1 )
        if self.lastMoveTo:
            self.lastMoveTo = (Ny - self.lastMoveTo[0] - 1, Nx - self.lastMoveTo[1] - 1)
        if self.moveFrom:
            self.moveFrom = (Ny - self.moveFrom[0] - 1, Nx - self.moveFrom[1] - 1)
        if self.moveTo:
            self.moveTo = (Ny - self.moveTo[0] - 1, Nx - self.moveTo[1] - 1)

    def flipH(self):
        self.board = (np.fliplr(self.board))
        if self.lastMoveFrom:
            self.lastMoveFrom = (self.lastMoveFrom[0], Nx-self.lastMoveFrom[1] -1 )
        if self.lastMoveTo:
            self.lastMoveTo = ( self.lastMoveTo[0] - 1, Nx - self.lastMoveTo[1] - 1)
        if self.moveFrom:
            self.moveFrom = ( self.moveFrom[0] , Nx - self.moveFrom[1] - 1)
        if self.moveTo:
            self.moveTo = (self.moveTo[0], Nx - self.moveTo[1] - 1)

    def printBoard(self):
        bs = self.board.shape
        for i, y in enumerate(range(bs[0])):
            print(i, end="")
            for x in range(bs[1]):
                print(chr(self.board[y,x]), end=" ")

            print("")
        print("last move from %s to %s" %  (self.moveFrom, self.moveTo))

    def getWinMove(self):
        # ret = archess.getMaxEatMove(self.board.tolist())
        #
        # if len(ret) == 3:
        #     if  ret[2] > 0:
        #         return [(ret[0][0], ret[0][1], ret[1][0], ret[1][1]), ret[2]]
        # return []
        #
        coords = [(a, b) for a in range(Ny) for b in range(Nx)]
        coords.sort(key=lambda c: (self.board == ord('K'))[c], reverse=True)
        posKing = coords[0]
        coords = [(a, b) for a in range(Ny) for b in range(Nx)]
        for moveFrom in coords:
            move = (moveFrom[0], moveFrom[1], posKing[0], posKing[1])
            if self.is_move_reasonable(move):
                return [move]

        return []

    def is_move_reasonable(self, move):
        fromX = move[1]
        fromY = move[0]
        toX = move[3]
        toY = move[2]

        ydiff = abs(fromY - toY)
        xdiff = abs(fromX - toX)
        dirY = None
        dirX = None
        if ydiff != 0:
            dirY = (toY - fromY) // ydiff
        if xdiff != 0:
            dirX = (toX - fromX) // xdiff
        src = self.board[(fromY, fromX)]
        tgt =self.board[(toY, toX)]
        if fromY == toY and fromX == toX:
            return False;
        if src < ord('a'):
            return False
        if tgt >= ord('a'):
            return False

        if src == ord('k'):
            if toX <3 or toX > 5:
                return False
            if tgt == ord('K') and toX == fromX:
                kingKill = True
                for along in range(fromY + 1, toY):
                    if self.board[along, fromX] != ord(' '):
                        kingKill = False
                        break;
                if kingKill:
                    return True

            if toY > 2 or abs(toY-fromY) + abs(toX-fromX) >1 :
                return False

        elif src == ord('r'):
            if fromX != toX and toY != fromY:
                return False
            if dirX is None:
                for indexY in range(fromY + dirY, toY, dirY):
                    if self.board[indexY, fromX] != ord(' '):
                        return False
            if dirY is None:
                for indexX in range(fromX + dirX, toX, dirX):
                    if self.board[fromY, indexX] != ord(' '):
                        return False
        elif src == ord('h'):
            if fromX == toX or toY == fromY:
                return False

            if xdiff + ydiff != 3:
                return False
            if ydiff == 2:
                if self.board[fromY + (toY - fromY) // 2, fromX] != ord(' '):
                    return False
            if xdiff == 2:
                if self.board[fromY, fromX + (toX - fromX) // 2] != ord(' '):
                    return False
        elif src == ord('c'):
            if fromX != toX and toY != fromY:
                return False
            obstacle = 0
            if dirX is None:
                for indexY in range(fromY+ dirY, toY, dirY):
                    if self.board[indexY, fromX] != ord(' '):
                        obstacle = obstacle +1
            if dirY is None :
                for indexX in range(fromX + dirX, toX, dirX):
                    if self.board[fromY, indexX] != ord(' '):
                        obstacle = obstacle + 1
            if obstacle > 1:
                return False
            if obstacle == 0 and tgt != ord(' '):
                return False
            if obstacle == 1 and tgt == ord(' '):
                return False
        elif src == ord('a'):
            if xdiff != 1 or ydiff != 1:
                return False
            if toY > 2 or toX < 3 or toX > 5:
                return False
        elif src == ord('e'):
            if xdiff != 2 or ydiff != 2:
                return False
            if toY > 4:
                return False
            if self.board[fromY + (toY - fromY) // 2, fromX + (toX - fromX) // 2] != ord(' '):
                return False
        elif src == ord('p'):
            if fromY > toY:
                return False
            if fromY <=4 and xdiff > 0:
                return False
            if xdiff + ydiff > 1:
                return False
        else:
            return False

        return True





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