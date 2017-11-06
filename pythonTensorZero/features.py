'''
Features used by AlphaGo, in approximate order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty  
Ones                    1 Constant plane of 1s 
    (Because of convolution w/ zero-padding, this is the only way the NN can know where the edge of the board is!!!)
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s

All features with 8 planes are 1-hot encoded, with plane i marked with 1 
only if the feature was equal to i. Any features >= 8 would be marked as 8.
'''

import numpy as np
import cc
from utils import product

# Resolution/truncation limit for one-hot features
P = 8

def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.uint8)
    capped = np.minimum(feature, planes)
    onehot_index_offsets = np.arange(0, product(onehot_features.shape), planes) + capped.ravel()
    # A 0 is encoded as [0,0,0,0], not [1,0,0,0], so we'll
    # filter out any offsets that are a multiple of $planes
    # A 1 is encoded as [1,0,0,0], not [0,1,0,0], so subtract 1 from offsets
    nonzero_elements = (capped != 0).ravel()
    nonzero_index_offsets = onehot_index_offsets[nonzero_elements] - 1
    onehot_features.ravel()[nonzero_index_offsets] = 1
    return onehot_features

def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

@planes(3)
def stone_color_feature(position):
    board = position.board
    features = np.zeros([cc.Ny, cc.Nx, 3], dtype=np.uint8)
    lowerFirst = ord('a')
    emptyPlayer= ord(' ')

    features[board == emptyPlayer, 2] = 1
    features[board >= lowerFirst, 0] = 1
    features[np.logical_and(board!=emptyPlayer, board<lowerFirst), 1] = 1

    return features

@planes(1)
def ones_feature(position):
    return np.ones([cc.Ny, cc.Nx, 1], dtype=np.uint8)

@planes(2) #only last from to
def recent_move_feature(position):
    onehot_features = np.zeros([cc.Ny, cc.Nx, 2], dtype=np.uint8)
    if position.lastMoveFrom is not None:
        onehot_features[position.lastMoveFrom, 0] = 1
        onehot_features[position.lastMoveTo, 1] = 1
    return onehot_features

@planes(P)
def liberty_feature(position):
    return make_onehot(position.get_liberties(), P)

@planes(2)
def winner_feature(position):
    ones = np.ones([cc.Ny, cc.Nx, 1], dtype=np.uint8)
    zeros = np.zeros([cc.Ny, cc.Nx, 1], dtype=np.uint8)
    if (position.win):
        return np.concatenate([ones, zeros], axis=2)
    else:
        return np.concatenate([zeros, ones], axis=2)

def valid(y,x):
    if y <0 or x <0 or y >=cc.Ny or x >=cc.Nx:
        return False;
    return True;
def fillPowerCover(board, loc, feature, plane):
    if len(loc) == 0:
        return

    piece = board[loc[0]]
    if piece >= ord('A') and  piece <= ord('Z'):
        piece = piece +ord('a') - ord('A')
    pieceWSide = board[loc[0]]
    for y,x in loc:
        if piece == ord('k') :
            posslist = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]
            for i,j in posslist:
                if valid(i, j) and (i <=2 or i >=7) and (j >=3 and j <=5):
                    feature[i,j, plane] = 1
        elif piece == ord('a') :
            posslist = [(y + 1, x+1), (y - 1, x-1), (y-1, x + 1), (y+1, x - 1)]
            for i, j in posslist:
                if valid(i,j) and (i <= 2 or i >= 7) and (j >= 3 and j <= 5):
                    feature[i, j, plane] = 1
        elif piece == ord('e'):
            posslist = [(y + 2, x + 2), (y - 2, x - 2), (y - 2, x + 2), (y + 2, x - 2)]
            for i, j in posslist:
                if valid(i, j) and \
                        ((y <= 4 and pieceWSide == piece) or (y > 4 and pieceWSide != piece)):
                    if board[(i+y) //2, (j+x)//2] == ord(' '):
                        feature[i, j, plane] = 1
        elif piece == ord('h'):
            posslist = [(y + 2, x + 1), (y + 2, x - 1), (y +1, x + 2), (y + 1, x - 2), (y - 2, x + 1), (y - 2, x - 1), (y -1, x + 2), (y - 1, x - 2)]
            for i, j in posslist:
                if valid(i, j):
                    if abs(i-y) ==2 and board[(i - y) // 2 +y, x] == ord(' '):
                        feature[i, j, plane] = 1
                    if abs(j-x) ==2 and board[y, (j - x) // 2 + x] == ord(' '):
                        feature[i, j, plane] = 1
        elif piece == ord('r'):
            for i in range(y-1, -1, -1):
                feature[i, x, plane] = 1
                if board[i, x] != ord(' '):
                    break;
            for i in range(y+1, cc.Ny):
                feature[i, x, plane] = 1
                if board[i, x] != ord(' '):
                    break;
            for i in range(x-1, -1, -1):
                feature[y, i, plane] = 1
                if board[y, i] != ord(' '):
                    break;
            for i in range(x+1, cc.Nx):
                feature[y, i, plane] = 1
                if board[y, i] != ord(' '):
                    break;
        elif piece == ord('c'):
            for i in range(y-1, -1, -1):
                if board[i, x] == ord(' '):
                    feature[i, x, plane] = 1
                else:
                    for j in range(i - 1, -1, -1):
                        if board[j, x] != ord(' '):
                            feature[j, x, plane] = 1
                            break;
                    break;
            for i in range(y+1, cc.Ny):
                if board[i, x] == ord(' '):
                    feature[i, x, plane] = 1
                else:
                    for j in range(i + 1, cc.Ny):
                        if board[j, x] != ord(' '):
                            feature[j, x, plane] = 1
                            break;
                    break;
            for i in range(x-1, -1, -1):
                if board[y, i] == ord(' '):
                    feature[y, i, plane] = 1
                else:
                    for j in range(i - 1, -1, -1):
                        if board[y, j] != ord(' '):
                            feature[y, j, plane] = 1
                            break;
                    break;
            for i in range(x+1, cc.Nx):
                if board[y, i] == ord(' '):
                    feature[y, i, plane] = 1
                else:
                    for j in range(i + 1, cc.Nx):
                        if board[y, j] != ord(' '):
                            feature[y, j, plane] = 1
                            break;
                    break;
        elif piece == ord('p') :
            if y <= 4 and pieceWSide == piece:
                feature[y + 1, x, plane] = 1
            elif y > 4 and pieceWSide != piece:
                feature[y - 1, x, plane] = 1
            else:
                if x != cc.Nx - 1:
                    feature[y, x + 1, plane] = 1
                if x != 0:
                    feature[y, x - 1, plane] = 1
                if y <= 4 and y != 0 :
                    feature[y - 1, x, plane] = 1
                elif y > 4 and y != cc.Ny -1 :
                    feature[y + 1, x, plane] = 1
        # elif piece == ord('e'):
        #     posslist = [(y + 2, x + 2), (y - 2, x - 2), (y - 2, x + 2), (y + 2, x - 2)]
        #     for i, j in posslist:
        #         if i >
        #         if pieceWSide == piece:
        #             if (i <= 2 or i >= 7) and (j >= 3 and j <= 5):
        #                 featurePlane[i, j] = 1


def fillPlanes4Piece(piece, board, features, plane):
    tgtMatrix = board == piece
    y, x = np.where(tgtMatrix)
    features[tgtMatrix, plane] = 1
    plane += 1
    fillPowerCover(board, list(zip(y, x)), features, plane)
    plane += 1
    return plane

@planes(28)
def piece_type_feature(position):
    features = np.zeros([cc.Ny, cc.Nx, 28], dtype=np.uint8)
    kingPiece = ord('k')
    ridePiece = ord('r')
    horsePiece = ord('h')
    canonPiece = ord('c')
    advisorPiece = ord('a')
    elePiece = ord('e')
    pawnPiece = ord('p')
    kingPiece2 = ord('K')
    ridePiece2 = ord('R')
    horsePiece2 = ord('H')
    canonPiece2 = ord('C')
    advisorPiece2 = ord('A')
    elePiece2 = ord('E')
    pawnPiece2 = ord('P')

    p = 0
    p = fillPlanes4Piece(kingPiece, position.board, features, p)
    p = fillPlanes4Piece(kingPiece2, position.board, features, p)
    p = fillPlanes4Piece(ridePiece, position.board, features, p)
    p = fillPlanes4Piece(ridePiece2, position.board, features, p)
    p = fillPlanes4Piece(horsePiece, position.board, features, p)
    p = fillPlanes4Piece(horsePiece2, position.board, features, p)
    p = fillPlanes4Piece(canonPiece, position.board, features, p)
    p = fillPlanes4Piece(canonPiece2, position.board, features, p)
    p = fillPlanes4Piece(advisorPiece, position.board, features, p)
    p = fillPlanes4Piece(advisorPiece2, position.board, features, p)
    p = fillPlanes4Piece(elePiece, position.board, features, p)
    p = fillPlanes4Piece(elePiece2, position.board, features, p)
    p = fillPlanes4Piece(pawnPiece, position.board, features, p)
    p = fillPlanes4Piece(pawnPiece2, position.board, features, p)
    assert p == 28

    return features

DEFAULT_FEATURES = [
    stone_color_feature,
    piece_type_feature,
    ones_feature,
#    recent_move_feature,
]

def extract_features(position, features=DEFAULT_FEATURES):
    return np.concatenate([feature(position) for feature in features], axis=2)

def bulk_extract_features(positions, features=DEFAULT_FEATURES):
    num_positions = len(positions)
    num_planes = sum(f.planes for f in features)
    output = np.zeros([num_positions, cc.Ny, cc.Nx, num_planes], dtype=np.uint8)
    moveFrom  = np.zeros([num_positions,cc.Ny, cc.Nx], dtype=np.uint8)
    moveTo = np.zeros([num_positions, cc.Ny, cc.Nx], dtype=np.uint8)

    result = np.zeros([num_positions, 2], dtype=np.uint8)

    for i, pos in enumerate(positions):
        output[i] = extract_features(pos, features=features)
        moveFrom[i, pos.moveFrom[0], pos.moveFrom[1]] = 1
        moveTo[i, pos.moveTo[0], pos.moveTo[1]] = 1
        result[i, 0] = int(pos.win)
        result[i, 1] = min(pos.step, 255)
    #encode move stuff
    return output, moveFrom, moveTo, result


features = np.zeros([cc.Ny, cc.Nx], dtype=np.uint8)
features[2,2] = 3
features[2,3] = 7

aaa = make_onehot(features, 8)
print(features)
print(aaa)