#include <iostream>
#include "Python.h"
#include <math.h>
#include <vector>
#include <sstream>
#include <list>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
using namespace std;

#define pk 1
#define pr 2
#define ph 3
#define pc 4
#define pa 5
#define pe 6
#define pp 7

#define invalid -100000000
int PieceScore[] = {0, 10000000, 10000, 5000, 5000, 2500, 2500, 1000};

class Move {
public:
	int x1 = -1;
	int y1;
	int x2;
	int y2;

	int ate;

	int score = 0;
	int sign;
	Move(int yy1, int xx1, int yy2, int xx2, int _sign):
		x1(xx1), y1(yy1), x2(xx2), y2(yy2){
		sign = _sign;
		ate = 0;
	}
	Move(){

	}
	list<Move> moveList;
	string toString() {
		ostringstream oss;
		oss << y1 << "," << x1 << " to " << y2 << "," << x2 << " score:" << score << " sign:" << sign ;
		return oss.str();
	}

	void addMove(Move& nextMove) {
		score += nextMove.score;
		moveList.insert(moveList.end(), nextMove.moveList.begin(), nextMove.moveList.end());
	}
	void finalizeMove() {
		moveList.push_back(*this);
	}

};
auto comp = [](const Move& lhs, const Move& rhs) {
	return lhs.score > rhs.score;
};
auto compOpp = [](const Move& lhs, const Move& rhs) {
	return lhs.score < rhs.score;
};

class Piece {
public:
	int x;
	int y;
	int piece;
	bool dead;
	Piece(int _y, int _x, int _piece):x(_x), y(_y), piece(_piece){
		dead = false;
	}
	string toString() {
		ostringstream oss;
		oss << "piece:" << piece << " loc: " << y << "," << x << " dead:" << dead;
		return oss.str();
	}
};

class Chess{
public:
	Chess(PyObject *args) :
		board(10, vector<int>(9)){
		posList.reserve(16);
		negList.reserve(16);
		PyObject *pList;
		PyObject *pItem;
		PyObject *pItemList;

//		Py_ssize_t n;
		int i;

		if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &pList)) {
		    PyErr_SetString(PyExc_TypeError, "parameter must be a list.");
		    return;
		}
		int ny = PyList_Size(pList);
		for (i=0; i<ny; i++) {
			pItemList = PyList_GetItem(pList, i);

		    if (!PyList_Check(pItemList))
		    {
				PyErr_SetString(PyExc_TypeError, "parameter must be a list of list.");
				return;
			}
		    int nx = PyList_Size(pItemList);
		    for (int j = 0; j < nx; ++j) {
		    	pItem = PyList_GetItem(pItemList, j);
//
		    	if(!PyLong_Check(pItem)) {
					PyErr_SetString(PyExc_TypeError, "list items must be integers.");
					return;
				}
				int v = (int ) PyLong_AsLong(pItem);
				if (v == ' ')
					board[i][j] = 0;
				else {
					int sign = 1;
					if (v < 'a') {
						sign = -1;
						v = v + 'a' - 'A';
					}
					switch (v) {
					case 'k':
						v = pk;
						break;
					case 'r':
						v = pr;
						break;
					case 'h':
						v = ph;
						break;
					case 'c':
						v = pc;
						break;
					case 'a':
						v = pa;
						break;
					case 'e':
						v = pe;
						break;
					case 'p':
						v = pp;
						break;
					default:
						PyErr_SetString(PyExc_TypeError, "invalid piece found in matrix");
						return;
					}
					int piece = v * sign;
					score += PieceScore[v] * sign;
					if (sign > 0) {
						posList.push_back(Piece(i,j, piece));
					}
					else {
						negList.push_back(Piece(i,j, piece));
					}
					board[i][j] = piece;
				}
		    }
		}

		valid = true;
	}
	bool isValid() {
		return valid;
	}
	~Chess() {
	}
	bool validMov(Move mov, bool allowSamePiece= false) {
		if (mov.x1 <0 || mov.x2 < 0 || mov.y1<0 ||mov.y2<0|| mov.x1>=9 || mov.x2>=9 || mov.y1>=10 || mov.y2>=10)
			return false;

		if (allowSamePiece || board[mov.y1][mov.x1] * board[mov.y2][mov.x2] <= 0 )
			return true; //same piece cannot overlap

		return false;
	}

	void apply(Move& move) {
		int p = board[move.y1][move.x1];
		move.ate = board[move.y2][move.x2];
		board[move.y2][move.x2] = p;
		board[move.y1][move.x1] = 0;
		if (move.ate * p >0){
			cout << "ERROR, CANNOT EAT SELF PIECE" << move.toString() << endl;
		}

		{
			vector<Piece>* pAteList = &posList;
			vector<Piece>* pMoveList = &negList;
			if (p >0) {
				swap(pAteList, pMoveList);
			}
			int oriScore = score;
			if (move.ate != 0) {
				int scoreChange =  PieceScore[abs(move.ate)];
				if (move.ate > 0)
					score -= scoreChange;
				else
					score += scoreChange;

				for (auto& piece : *pAteList){
					if (piece.y == move.y2 && piece.x == move.x2) {
						piece.dead = true;
						break;
					}
				}
			}
			for (auto& piece : *pMoveList){
				if (piece.y == move.y1 && piece.x == move.x1) {
					piece.y = move.y2;
					piece.x = move.x2;
					break;
				}
			}
			move.score = score - oriScore;
		}

		//cout << setfill('0') << setw(2) <<  "after move"<< endl << toString() <<endl;

	}
	void rollback(Move& move) {
		int p = board[move.y2][move.x2];
		board[move.y2][move.x2] = move.ate;
		board[move.y1][move.x1] = p;
		vector<Piece>* pAteList = &posList;
		vector<Piece>* pMoveList = &negList;

		if (p >0) {
			swap(pAteList, pMoveList);
		}

		for (auto& piece : *pMoveList){
			if (piece.y == move.y2 && piece.x == move.x2) {
				piece.y = move.y1;
				piece.x = move.x1;
				break;
			}
		}
		if (move.ate != 0) {
			int scoreChange =  PieceScore[abs(move.ate)];
			if (move.ate > 0)
				score += scoreChange;
			else
				score -= scoreChange;

			for (auto& piece : *pAteList){
				if (piece.piece ==move.ate  && piece.dead ) {
					piece.dead = false;
					piece.y = move.y2;
					piece.x = move.x2;
					break;
				}
			}
		}
		//cout << "after rollback"<< endl << toString() <<endl;
	}
	list<Move> getMoves(Piece piece, int sign, int level) {
		list<Move> ret;
		if (piece.piece * sign < 0) {
			cout << "error in logic!!!!! sign wrong!" << endl;
		}
		int v = abs(piece.piece);
		Move mov(piece.y, piece.x, 0, 0, sign);
		switch (v) {
		case pk:
			for (int i = 0; i < 2; ++i) {
				int j = 1 - i;
				mov.y2 = piece.y +i;
				mov.x2 = piece.x +j;
				if (validMov(mov) && (mov.y2 <=2 || mov.y2 >=7) && (mov.x2 >=3 && mov.x2 <=5)){
					ret.push_back(mov);
				}
				mov.y2 = piece.y -i;
				mov.x2 = piece.x -j;
				if (validMov(mov) && (mov.y2 <=2 || mov.y2 >=7) && (mov.x2 >=3 && mov.x2 <=5)){
					ret.push_back(mov);
				}
			}
			for ( int i = piece.y + sign; i >=0 && i <10; i = i + sign) {
				if (board[i][piece.x] == 0)
					continue;
				if (abs(board[i][piece.x]) == pk) {
					mov.y2 = i;
					mov.x2 = piece.x;
					ret.push_back(mov);
					return ret;
				}
				break;
			}
			break;
		case pa:
			for (int i = -1; i < 2; i+=2) {
				mov.y2 = piece.y +i;
				mov.x2 = piece.x +i;
				if (validMov(mov) && (mov.y2 <=2 || mov.y2 >=7) && (mov.x2 >=3 && mov.x2 <=5)){
					ret.push_back(mov);
				}
				mov.y2 = piece.y +i;
				mov.x2 = piece.x -i;
				if (validMov(mov) && (mov.y2 <=2 || mov.y2 >=7) && (mov.x2 >=3 && mov.x2 <=5)){
					ret.push_back(mov);
				}
			}
			break;
		case pe:
			for (int i = -2; i < 3; i+=4) {
				mov.y2 = piece.y +i;
				mov.x2 = piece.x +i;
				if (validMov(mov) && ((mov.y2 <=4 && mov.y1 <=4) || (mov.y2 >4 && mov.y1 >4))){
					ret.push_back(mov);
				}
				mov.y2 = piece.y +i;
				mov.x2 = piece.x -i;
				if (validMov(mov) && ((mov.y2 <=4 && mov.y1 <=4) || (mov.y2 >4 && mov.y1 >4))){
					ret.push_back(mov);
				}
			}
			break;
		case pp:

			mov.y2 = piece.y + sign;
			mov.x2 = piece.x;
			if (validMov(mov))
				ret.push_back(mov);

			if ((mov.y1 >4 && sign == 1) || (mov.y1 <=4 && sign == -1)) {
				mov.y2 = piece.y ;

				mov.x2 = piece.x + 1;
				if (validMov(mov))
					ret.push_back(mov);

				mov.x2 = piece.x - 1;
				if (validMov(mov))
					ret.push_back(mov);
			}
			break;
		case ph:
			for (int i = -1; i <= 2; i+=2) {
				for (int j = -1; j <= 2; j+=2){
					mov.y2 = mov.y1 + 2 * i;
					mov.x2 = mov.x1 + 1 * j;
					if (validMov(mov) && board[mov.y2-i][mov.x1] == 0) {
						ret.push_back(mov);
					}

					mov.y2 = mov.y1 + 1 * i;
					mov.x2 = mov.x1 + 2 * j;
					if (validMov(mov) && board[mov.y1][mov.x2 -j] == 0) {
						ret.push_back(mov);
					}

				}
			}
			break;
		case pr:
			mov.y2 = mov.y1;
			for ( int i = mov.x1 -1; i >= 0; --i){
				mov.x2 = i;
				if (!validMov(mov))
					break;
				ret.push_back(mov);
				if (board[mov.y2][mov.x2] != 0)
					break;
			}
			for ( int i = mov.x1 +1; i < 9; ++i){
				mov.x2 = i;
				if (!validMov(mov))
					break;
				ret.push_back(mov);
				if (board[mov.y2][mov.x2] != 0)
					break;
			}
			mov.x2 = mov.x1;
			for ( int i = mov.y1 -1; i >= 0; --i){
				mov.y2 = i;
				if (!validMov(mov))
					break;
				ret.push_back(mov);
				if (board[mov.y2][mov.x2] != 0)
					break;
			}
			for ( int i = mov.y1 +1; i < 10; ++i){
				mov.y2 = i;
				if (!validMov(mov))
					break;
				ret.push_back(mov);
				if (board[mov.y2][mov.x2] != 0)
					break;
			}
			break;
		case pc:
			mov.y2 = mov.y1;
			int obstacle = 0;
			for ( int i = mov.x1 -1; i >= 0; --i){
				mov.x2 = i;
				if (pcChecker(mov, obstacle, ret))
					break;
			}
			obstacle = 0;
			for ( int i = mov.x1 +1; i < 9; ++i){
				mov.x2 = i;
				if (pcChecker(mov, obstacle, ret))
					break;
			}
			mov.x2 = mov.x1;
			obstacle = 0;
			for ( int i = mov.y1 -1; i >= 0; --i){
				mov.y2 = i;
				if (pcChecker(mov, obstacle, ret))
					break;
			}
			obstacle = 0;
			for ( int i = mov.y1 +1; i < 10; ++i){
				mov.y2 = i;
				if (pcChecker(mov, obstacle, ret))
					break;
			}
			break;

		}

		return ret;
	}
	bool pcChecker(Move& mov, int& obstacle, list<Move>& ret)
	{
		if (!validMov(mov, true))
			return true;
		if (obstacle == 0){
			if (board[mov.y2][mov.x2] == 0)
				ret.push_back(mov);
			else {
				obstacle++;
			}
		}
		else {
			if (board[mov.y2][mov.x2] * board[mov.y1][mov.x1]  <0)
			{
				ret.push_back(mov);
				return true;
			}
		}
		return false;
	}

	Move getMaxScoreMove( int sign, int level) {
		vector<Move> scoredMove;
		scoredMove.reserve(160);
		vector<Piece> * ptrList = &posList;
		if (sign < 0)
			ptrList = &negList;
		for (auto piece : *ptrList) {
			if (piece.dead) continue;
			list<Move> possibleMove = getMoves(piece, sign, level); //caled the score inside the move

			for (auto move : possibleMove) {
				if (board[move.y1][move.x1] * board[move.y2][move.x2] >= 0)
					continue;
//				if (level == 0)
//					cout << " first play possible move:" << move.toString() << endl;
				if (abs(board[move.y2][move.x2]) ==pk){
					move.score = sign * PieceScore[pk];
					move.finalizeMove();
					return move;
				}
				int lastScore = move.score;
				vector<Piece> posTmpList = posList;
				vector<Piece> negTmpList = negList;
				apply(move);
				Move tmpMove = move;
				{
					Move nextMove = getMaxScoreMove(-sign, level + 1); //actual score
					if (abs(nextMove.score) != PieceScore[pk] ) { // must not allow other to kill own king
						tmpMove.addMove(nextMove);
						if (tmpMove.score* sign > 0)
							scoredMove.push_back(tmpMove);
					}
				}
				rollback(move);
				posList = posTmpList;
				negList = negTmpList;
				move.score = lastScore;
			}
		}

		if (scoredMove.empty()) {
			return Move();
		}
		if (sign > 0) {
			sort(scoredMove.begin(), scoredMove.end(), comp);
		}
		else {
			sort(scoredMove.begin(), scoredMove.end(), compOpp);
		}
		int index = 0;
		int maxScore = scoredMove[0].score;

		for (auto m : scoredMove) {
			if (m.score != maxScore)
				break;
			++index;
		}
		//[0,index) are equal score, randome one out
		Move ret = scoredMove[rand()%index];
		ret.finalizeMove();

		if (level == 0){
			cout << "chosen:" <<  ret.y1 << "," << ret.x1 << " to " << ret.y2 << "," << ret.x2 << " :score:" << ret.score << endl;
//			cout << "move path" << endl;
//			for (auto m : ret.moveList) {
//				cout << m.toString() <<endl;
//			}
		}
		return ret;

	}


	string toString() {
		ostringstream oss;
		for ( int i = 0; i < 10; ++i ) {
			for (int j = 0; j < 9; ++j) {
				oss << board[i][j] << " ";
			}
			oss << endl;
		}
		oss << "posList:" << endl;
		for (auto ppp: posList) {
			oss << ppp.toString() << endl;
		}
		oss << "negList:" << endl;
		for (auto ppp: negList) {
			oss << ppp.toString() << endl;
		}
		return oss.str();
	}
private:
	bool valid = false;
	vector< vector<int>  > board;
	vector<Piece> posList;
	vector<Piece> negList;
	int score;
};
static PyObject *archessError;
int Prime(int n) {
	if (n <=1){
		cout << "Not Prime." << endl;
		return -1;
	}
	if (n <=3) {
		cout << "Prime" << endl;
		return 1;
	}
	for (int i =2 ; i*i <=n; ++i) {
		if (n%i ==0) {
			cout << "Not prime" << endl;
			return  -1;
		}
	}
	cout << "PRIME" << endl;
	return 1;
}


extern "C"
{
static PyObject* archess_Prime(PyObject *self, PyObject *args) {
	int n =0 ;
	if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
	return Py_BuildValue("i", Prime(n));
}
static PyObject* getMaxEatMove(PyObject *self, PyObject *args) {
	int n =0 ;
	Chess chess(args);

	if (chess.isValid()){
		Move m = chess.getMaxScoreMove(1, 0);

		PyObject *rslt = PyTuple_New(3);

		PyObject *score, *loc1, *loc2;
		loc1 = PyTuple_New(2);
		loc2 = PyTuple_New(2);

		PyTuple_SetItem(loc1, 0, Py_BuildValue("i", m.y1));
		PyTuple_SetItem(loc1, 1, Py_BuildValue("i", m.x1));
		PyTuple_SetItem(loc2, 0, Py_BuildValue("i", m.y2));
		PyTuple_SetItem(loc2, 1, Py_BuildValue("i", m.x2));

		score = Py_BuildValue("i", m.score);
		PyTuple_SetItem(rslt, 0, loc1);
		PyTuple_SetItem(rslt, 1, loc2);
		PyTuple_SetItem(rslt, 2, score);
		return rslt;

	}

	return PyTuple_New(0);
}


static PyMethodDef archessMethods[]= {
		{"archess_Prime", (PyCFunction) archess_Prime, METH_VARARGS},
		{"getMaxEatMove", (PyCFunction) getMaxEatMove, METH_VARARGS},
		{NULL, NULL, 0, NULL}
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "archess", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    archessMethods
};

PyMODINIT_FUNC PyInit_archess() {
	//Py_InitModule("archess", archessMethods);
	srand(std::time(0));

	return PyModule_Create(&cModPyDem);
}
}
