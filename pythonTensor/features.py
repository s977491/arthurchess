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


@planes(7)
def piece_type_feature(position):
    features = np.zeros([cc.Ny, cc.Nx, 7], dtype=np.uint8)
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

    features[np.logical_or(position.board == kingPiece, position.board == kingPiece2), 0] = 1
    features[np.logical_or(position.board == ridePiece, position.board == ridePiece2), 1] = 1
    features[np.logical_or(position.board == horsePiece, position.board == horsePiece2), 2] = 1
    features[np.logical_or(position.board == canonPiece, position.board == canonPiece2), 3] = 1
    features[np.logical_or(position.board == advisorPiece, position.board == advisorPiece2), 4] = 1
    features[np.logical_or(position.board == elePiece, position.board == elePiece2), 5] = 1
    features[np.logical_or(position.board == pawnPiece, position.board == pawnPiece2), 6] = 1

    return features

DEFAULT_FEATURES = [
    stone_color_feature,
    piece_type_feature,
    ones_feature,
    recent_move_feature,
    winner_feature,
]

def extract_features(position, features=DEFAULT_FEATURES):
    return np.concatenate([feature(position) for feature in features], axis=2)

def bulk_extract_features(positions, features=DEFAULT_FEATURES):
    num_positions = len(positions)
    num_planes = sum(f.planes for f in features)
    output = np.zeros([num_positions, cc.Ny, cc.Nx, num_planes], dtype=np.uint8)
    move = np.zeros([num_positions, cc.Ny,cc.Nx, cc.Ny, cc.Nx], dtype=np.uint8)
    result = np.zeros([num_positions, 2], dtype=np.uint8)

    for i, pos in enumerate(positions):
        output[i] = extract_features(pos, features=features)
        move[i, pos.moveFrom[0], pos.moveFrom[1], pos.moveTo[0], pos.moveTo[1]] = 1
        result[i, 0] = int(pos.win)
        result[i, 1] = min(pos.step, 255)
    #encode move stuff
    return output, move.reshape(num_positions, cc.Ny*cc.Nx* cc.Ny*cc.Nx), result


features = np.zeros([cc.Ny, cc.Nx], dtype=np.uint8)
features[2,2] = 3
features[2,3] = 7

aaa = make_onehot(features, 8)
print(features)
print(aaa)