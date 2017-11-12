import gzip
import copy
import math
import random
import sys
import time
import cc
import numpy as np
import struct
import features
import utils
from load_data_sets import DataSet, parse_data_sets
from contextlib import contextmanager

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))

CHUNK_HEADER_FORMAT = "iiii?"

c_PUCT = 3
eee = 0.5


class MCTSNode():
    '''
    A MCTSNode has two states: plain, and expanded.
    An plain MCTSNode merely knows its Q + U values, so that a decision
    can be made about which MCTS node to expand during the selection phase.
    When expanded, a MCTSNode also knows the actual position at that node,
    as well as followup moves/probabilities via the policy network.
    Each of these followup moves is instantiated as a plain MCTSNode.
    '''
    temper = False
    @staticmethod
    def root_node(position, move_probabilities):
        node = MCTSNode(None, None, 0)
        node.position = position
        #artprob = np.ones([cc.Ny, cc.Nx, cc.Ny, cc.Nx], dtype=np.float32) / cc.Ny / cc.Nx / cc.Ny / cc.Nx
        #noiseFn = np.random.dirichlet((3, 97), cc.Ny*cc.Nx*cc.Ny* cc.Nx)[:,0].reshape(cc.Ny, cc.Nx, cc.Ny, cc.Nx)
        #artprob += noiseFn
        node.expand(move_probabilities)
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        if MCTSNode.temper:
            noiseFn = np.random.dirichlet((3, 97), 1)
            self.prior = self.prior * (1 - eee) + eee * noiseFn[0][0]
        else:
            self.prior = self.prior

        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0 # average of all outcomes involving this node
        self.N = 0 # number of times node was visited
        self.calU()  # monte carlo exploration bonus

        self.W = 0
        self.done = False
        #self.estV = 0

    def __repr__(self):
        return "<MCTSNode(%s) move=%s prior=%s score=%s is_expanded=%s>" % (self.N, self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        if not MCTSNode.temper:
            #noiseFn = np.random.dirichlet((3, 97), 1)
            return self.Q  + self.U

        return self.Q + self.U
    def is_expanded(self):
        return self.position is not None

    def compute_position(self):
        self.position = self.parent.position.clone()
        self.position.move((self.move[0], self.move[1]), (self.move[2], self.move[3]))
        self.position.flip()
        return self.position

    def expand(self, move_probabilities):
        won = 1
        possMov = self.position.getWinMove()
        if not possMov:
            possMov = self.position.possibleMoves()
            won = 0

            #check if same position ever and prevent repeating move
            if self.parent is None or self.parent.parent is None:
                pass
            else:
                checkRoot = self.parent.parent #previous move must be different, skip check
                for layers in range(25): #make sure 25 layers no repeat pattern only
                    if checkRoot.parent is None or checkRoot.parent.parent is None:
                        break
                    checkMove = checkRoot.parent.move
                    checkRoot = checkRoot.parent.parent

                    if np.array_equal(checkRoot.position.board, self.position.board):
                        if checkMove in possMov:
                            possMov.remove(checkMove)

        if not possMov:
            print("Unexpected no move left, concced")
            won = 2
        sumProb = 0.0
        probMap = {}

        if won == 1:
            probMap[possMov[0]] = 1
            sumProb = 1

        elif won == 0:
            for mov in possMov:
                sumProb += move_probabilities[mov]
                probMap[mov] = move_probabilities[mov]
        else:
            self.done = 2
            return 2
        if sumProb == 0:
            #print ("unexpected prob sum = 0")
            for mov in possMov:
                sumProb += 0.001
                probMap[mov] = 0.001
        samples = len(probMap)

        for move, prob in probMap.items():
            pp = prob / sumProb

            self.children[move] = MCTSNode(self, move, pp)

        if won == 1:
            self.children[move].done = won


        return won
    def calU(self):


        self.U = c_PUCT * (self.prior) / (1 + self.N)

    def backup_valueImpl(self, node, value):
        node.N += 1
        node.W += value
        node.Q = node.W / node.N

        if node.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.

        self.calU()

    def backup_value(self, value):
        node = self
        while not node is None:
            self.backup_valueImpl(node, value)
            # must invert, because alternate layers have opposite desires
            value = -value
            node = node.parent

    def select_leaf(self):
        current = self
        if current.done >0:
            print ("strange that root already done!, no solution!")
            return current
        while current.is_expanded():
            current = max(current.children.values(), key=lambda node: node.action_score)
            if current.done> 0:
                print("select should avoid this done node, unexpected")
                return current
        return current


class MCTSPlayerMixinTrainer:
    def __init__(self, policy_network, seconds_per_move=5):
        self.policy_network = policy_network
        self.seconds_per_move = seconds_per_move
        self.max_rollout_depth = cc.Ny * cc.Nx * 3
        super().__init__()

    def suggest_move(self, position):
        start = time.time()
        yv, move_probs = self.policy_network.run(position)
        root = MCTSNode.root_node(position, move_probs, yv)
        while time.time() - start < self.seconds_per_move:
            self.tree_search(root)
        print("Searched for %s seconds" % (time.time() - start), file=sys.stderr)
        sorted_moves = sorted(root.children.keys(), key=lambda move, root=root: root.children[move].N, reverse=True)
        for move in sorted_moves:
            if position.is_move_reasonable(move):
                return move
        return None

    def trainning(self, gameNo):
        won = 0
        position = cc.get_start_board()
        yv, move_probs = self.policy_network.run(position)
        root = MCTSNode.root_node(position, move_probs)
        curRoot = root
        winMove = None
        side = 1
        for step in range(300):
            winMove = position.getWinMove()
            if winMove:
                position.printBoard()
                print("Game End, side %d win" % (step %2))
                won = 1
                break
            start = time.time()
            if step > 3:
                MCTSNode.temper = True

            loop = 0
            hintNode = None
            #while time.time() - start < self.seconds_per_move:
            with timer("treeSearching"):
                for loop in range(200):
                    hintNode = self.tree_search(curRoot, hintNode)
                    #loop += 1

            print ("MCTS run %d" % (loop))
            if len(curRoot.children) ==0:
                position.printBoard()
                print("error , no move somehow, conceed")
                won = 2
                break
            sorted_moves = sorted(curRoot.children.keys(), key=lambda move, curRoot=curRoot: curRoot.children[move].N, reverse=True)

            nextMove = sorted_moves[0]
            position.move((nextMove[0], nextMove[1]), (nextMove[2], nextMove[3]) )

            if side ==1:
                position.printBoard()
            print("Game: %d Step %d" %(gameNo, step))

            position.flip()
            if side == -1:
                position.printBoard()
            side = -side

            #free memory of nephrew
            for broMove, broNode in curRoot.children.items():
                if broMove != nextMove: #clear it
                    broNode.children.clear()
            curRoot = curRoot.children[nextMove]


        if not curRoot.is_expanded():
            self.tree_search(curRoot)

        if won == 0:
            curRoot.backup_value(0)
            curRoot.Q = 0

        self.tree_train(curRoot)
        #train the network now

    def tree_train(self, lastRoot):
        print("tree train", file=sys.stderr)

        featureArr = []
        yVArr = []
        yMovArr =[]

        while lastRoot is not None:
            if not lastRoot.children:
                print ("unexpected no children root!")
            else:
                yv = np.zeros([1], dtype=np.float32)
                yv[0] = lastRoot.Q

                totalQ = 0

                pos1 = lastRoot.position
                moveArr = np.zeros([cc.Ny, cc.Nx, cc.Ny, cc.Nx], dtype=np.float32)
                for move, node in lastRoot.children.items():
                    totalQ += node.Q
                    moveArr[move] = node.Q

                moveArr = moveArr / totalQ
                yMovArr.append(moveArr.ravel())
                featureArr.append(features.extract_features(pos1))

                yVArr.append(yv)

                moveArr = np.zeros([cc.Ny, cc.Nx, cc.Ny, cc.Nx], dtype=np.float32)
                pos2 = lastRoot.position.clone()
                pos2.flipH()
                for move, node in lastRoot.children.items():
                    moveArr[move[0],cc.Nx-1- move[1], move[2],cc.Nx-1- move[3]] = node.Q

                moveArr = moveArr / totalQ
                yMovArr.append(moveArr.ravel())
                featureArr.append(features.extract_features(pos2))
                yVArr.append(yv)
                # vertical
                moveArr = np.zeros([cc.Ny, cc.Nx, cc.Ny, cc.Nx], dtype=np.float32)
                pos3 = lastRoot.position.clone()
                pos3.flipV()
                for move, node in lastRoot.children.items():
                    moveArr[cc.Ny-1-move[0], move[1], cc.Ny-1-move[2], move[3]] = node.Q

                moveArr = moveArr / totalQ
                yMovArr.append(moveArr.ravel())
                featureArr.append(features.extract_features(pos3))
                yVArr.append(yv)

                #vertical + horib
                moveArr = np.zeros([cc.Ny, cc.Nx, cc.Ny, cc.Nx], dtype=np.float32)
                pos4 = lastRoot.position.clone()
                pos4.flipH()
                pos4.flipV()
                for move, node in lastRoot.children.items():
                    moveArr[cc.Ny-1-move[0],cc.Nx-1- move[1], cc.Ny-1-move[2],cc.Nx-1- move[3]] = node.Q

                moveArr = moveArr / totalQ
                yMovArr.append(moveArr.ravel())
                featureArr.append(features.extract_features(pos4))
                yVArr.append(yv)

            lastRoot = lastRoot.parent

        training_datasets = DataSet(np.array(featureArr), np.array(yMovArr), np.array(yVArr))


        with timer("training"):
            for i in range(3):
                self.policy_network.train(training_datasets)
                training_datasets.shuffle()

        self.policy_network.save_variables()

    def tree_search(self, root, hintNode):
        #print("tree search", file=sys.stderr)
        # selection
        useHint = False
        node = hintNode
        hintNodeDepth = -1
        justLost = False
        if hintNode is not None and len(hintNode.children) == 1:
            justLost = True
        while not node is None:
            hintNodeDepth +=1
            if node == root:
                useHint = True
                break
            # if justLost:
            #     print("JustLost Move parents mv %s" % [node.move])
            node = node.parent
        if useHint and hintNodeDepth >20:
            searchNode=hintNode
            for depth in range(hintNodeDepth):
                searchNode = searchNode.parent
            chosen_leaf = searchNode.select_leaf()
        else:
            chosen_leaf = root.select_leaf()
        # expansion
        if chosen_leaf.done ==1 :
            chosen_leaf.backup_value(1)
            return None
        elif chosen_leaf.done ==2 :
            chosen_leaf.backup_value(0)
            return None

        position = chosen_leaf.compute_position()
        if position is None:
            print("illegal move!", file=sys.stderr)
            # See go.Position.play_move for notes on detecting legality
            del chosen_leaf.parent.children[chosen_leaf.move]
            return
        #print("Investigating following position:\n%s" % (chosen_leaf.position,), file=sys.stderr)
        #move_probs = self.policy_network.run(position)
        yv, move_probs = self.policy_network.run(position)

        won = chosen_leaf.expand(move_probs)
        if won == 1:
            yv = -1
        elif won == 2:
            yv = 0
        else:
            pass
        # evaluation
        # value = self.estimate_value(root, chosen_leaf)
        # backup
        #print("value: %s" % yv, file=sys.stderr)
        # if MCTSNode.temper:
        #     noiseFn = (random.randint(0, 10000) -5000)/ 10000
        #     yv = yv * (1- eee) + (eee * noiseFn[0][0]
        chosen_leaf.backup_value(yv)
        return chosen_leaf

    # def estimate_value(self, root, chosen_leaf):
    #     # Estimate value of position using rollout only (for now).
    #     # (TODO: Value network; average the value estimations from rollout + value network)
    #     leaf_position = chosen_leaf.position
    #     current = copy.deepcopy(leaf_position)
    #     simulate_game(self.policy_network, current)
    #     print(current, file=sys.stderr)
    #
    #     perspective = 1 if leaf_position.to_play == root.position.to_play else -1
    #     return current.score() * perspective

