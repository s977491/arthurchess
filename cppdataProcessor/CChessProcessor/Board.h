#pragma once

using namespace std;

#define SIDE_RED "red"
#define SIDE_BLACK "black"
#define MX_X 9
#define MX_Y 10

#define isdebug false

class Location {
public:
	int x;
	int y;
	Location(int _y, int _x) : x(_x), y(_y) {}
	Location() { x = -1; y = -1; }

	bool valid() { return x >= 0 && x < MX_X && y >= 0 && y < MX_Y; }
	bool operator < (const Location& rhs) const {
		if (x == rhs.x) {
			return y < rhs.y;
		}
		return x < rhs.x;
	}
	bool operator ==(const Location& rhs) const {
		return x == rhs.x && y == rhs.y;
	}
	Location flip() {
		return Location(MX_Y-1 - y, MX_X-1 - x);
	}

};
std::ostream &operator<<(std::ostream &os, Location const &m);

enum Direction { NORMAL, DIAGONAL, HORSE };
class Board {
public:
	map<char, Direction> dirMap = { { 'r',NORMAL },{ 'h',HORSE },{ 'e',DIAGONAL },{ 'a',DIAGONAL },{ 'k',NORMAL },{ 'p',NORMAL },{ 'c',NORMAL } };
	//list<Location> fromList;
	//list<Location> toList;
	Location lastStepFrom, lastStepTo;
	ofstream& fp;
	string gameName;
	Board(ofstream& file, string& _gameName) : fp(file), gameName(_gameName){
		resetBoard();
	}
	virtual ~Board() {
	}
	string toString() {
		//print the board, previous move, and target move
	}
	void resetBoard() {
		pCurMap = &pieceMap;
		pOppMap = &pieceMapOpp;
		for (int i = 0; i < 2; ++i) {
			putrow(0, "rheakaehr");
			putrow(1, "         ");
			putrow(2, " c     c ");
			putrow(3, "p p p p p");
			putrow(4, "         ");
			flipSide();
		}
		//putrow(6, "P P P P P");
		//putrow(7, " C     C ");
		//putrow(9, "RHEAKAEHR");
	}
	void flipSide() {
		swap(position, positionOpp);
		swap(pCurMap, pOppMap);
		swap(curSide, oppSide);
	}
	void printBoard() {
		cout << "board now of RED" << endl;
		for (int y = 0; y < MX_Y; ++y) {
			for (int x = 0; x < MX_X; ++x) {
				cout << (arrposition)[y][x];
			}
			cout << endl;
		}
		cout << "----------------------------" << endl;

		for (auto entry : (pieceMap)) {
			cout << "piece " << entry.first << ": ";
			for (auto loc : entry.second) {
				cout << loc << " || ";
			}
			cout << endl;
		}
		cout << "----------------------------" << endl;

		for (auto entry : (pieceMapOpp)) {
			cout << "piece " << entry.first << ": ";
			for (auto loc : entry.second) {
				cout << loc << " || ";
			}
			cout << endl;
		}

		cout << "============================================" << endl;

	}
	void process(const vector<string>& data) {
		if (starter.empty()) {
			starter = data[2];
			
		}
		moveMap[data[2]].push_back(data[3]);
	}
	void writeFile(const Location& from, const Location& to) {
		if (lastStepFrom.valid()) {
			fp << lastStepFrom.x << " " << lastStepFrom.y << " " << lastStepTo.x << " " << lastStepTo.y << endl;
		}
		else {
			fp << endl;
		}
		fp << from.x << " " << from.y << " " << to.x << " " << to.y << endl;
		fp << (winner == curSide) << " " << totalStep << endl;
		for (int y = 0; y < MX_Y; ++y) {
			for (int x = 0; x < MX_X; ++x) {
				fp << (*position)[y][x];
			}
			fp << endl;
		}		
	}
	bool doMove(const string& move) {
		if (isdebug)
			cout << "going to move" << move << endl;
		char piece = (char) tolower(move[0]);
		auto pieceXPos = move[1];
		auto pieceDir = move[2];
		auto pieceDes = move[3];

		//check if exist
		auto& pieceSet = (*pCurMap)[piece];
		if (pieceSet.empty()) {
			cout << "cannot find the piece for " << move << endl;
			return false;
		}

		Location loc;
		Location locTgt;



		if (pieceXPos >= '1' && pieceXPos <= '9') {
			int poxX = pieceXPos - '1';
			auto iter = pieceSet.begin();
			for (; iter != pieceSet.end(); ++iter) {
				if (iter->x == poxX) {
					loc = *iter;
					pieceSet.erase(iter);
					break;
				}
			}			
		}
		else if (pieceXPos == '+' || pieceXPos == '-')
		{
			if (piece != 'p') {

				auto iter = pieceSet.begin();
				if (pieceXPos == '+') {
					iter = pieceSet.end();
					--iter;					
				}
				loc = *(iter);
				pieceSet.erase(iter);
			}
			else {
				auto iterLast = pieceSet.begin();
				auto iter = iterLast;
				iter++;
				for (; iter != pieceSet.end(); ++iter) {
					if (iter->x == iterLast->x) {
						if (pieceXPos == '-') {
							iter = iterLast;
						}
						loc = *iter;
						pieceSet.erase(iter);
						break;
					}
					iterLast = iter;
				}
			}

		}
		else {
			cout << "unexpected move : " << move << endl;
			return false;
		}

		//found location or not
		if (!loc.valid()) {
			cout << "fail to find piece for " << move << endl;
			return false;
		}
		auto pieceDirType = dirMap[piece];

		switch (pieceDir) {
		case '.':
			locTgt.x = (pieceDes - '1');
			locTgt.y = loc.y;
			break;
		case '+':
		case '-':
			switch (pieceDirType) {
			case NORMAL:
				locTgt.x = loc.x;
				if (pieceDir == '+') {
					locTgt.y = loc.y + (pieceDes - '0');
				}
				else {
					locTgt.y = loc.y - (pieceDes - '0');
				}
				break;
			case HORSE:
			case DIAGONAL:
				locTgt.x = pieceDes - '1';
				int shift = abs(locTgt.x - loc.x);

				int sign = 1;
				if (pieceDir == '-')
					sign = -1;
				if (pieceDirType == DIAGONAL) {
					locTgt.y = loc.y + shift * sign;
				}
				else {
					shift = 3 - shift;
					locTgt.y = loc.y + shift * sign;
				}
			}
			break;

		default:
			cout << "fail to find pieceDir for " << move << endl;
			return false;
			break;
		}
		if (isdebug)
			cout << "From:" << loc << "\t TO:" << locTgt << endl;
		auto flipLoc = loc.flip();
		auto flipLocTgt = locTgt.flip();

		//write to file
		writeFile(loc, locTgt);
		lastStepFrom = flipLoc;
		lastStepTo = flipLocTgt;

		(*position)[loc.y][loc.x] = ' ';
		(*positionOpp)[flipLoc.y][flipLoc.x] = ' ';

		char* p = &((*position)[locTgt.y][locTgt.x]);
		if (*p != ' ') {
			//eat a piece, need to remove the map stuff
			(*pOppMap)[tolower(*p)].erase(flipLocTgt);
		}
		*p = piece;
		(*positionOpp)[flipLocTgt.y][flipLocTgt.x] = toupper(piece);
		//add back the map of current
		(*pCurMap)[piece].insert(locTgt);

		if (isdebug)
			printBoard();

		flipSide();
		return true;
	}
	void finish() {
		curSide = starter;
		if (curSide == SIDE_RED)
			oppSide = SIDE_BLACK;
		else
			oppSide = SIDE_RED;
		auto& blackMoveList = moveMap[SIDE_BLACK];
		auto& redMoveList = moveMap[SIDE_RED];
		auto pList = &redMoveList;
		auto pOppList = &blackMoveList;
		if (starter == SIDE_BLACK)
			swap(pList, pOppList);
		totalStep = (*pList).size();
		if ((*pList).size() > (*pOppList).size()) {
			winner = starter;
		}
		else {
			winner = oppSide;
		}

		do {
			if (pList->empty())
				break;
			string move = pList->front();
			if (!doMove(move)) {
				cout << "serious error:" << gameName <<  endl;
				break;
			}
			pList->pop_front();

			swap(pList, pOppList);
		} while (true);
	}

private:
	char flip(char c) {
		if (c == ' ')
			return c;
		return (c - 65 + 32) % 64 + 65;
	}
	void putrow(int row, string line) {
		int x = 0;
		for (auto c : line) {
			(*position)[row][x] = c;
			if (c != ' ') {
				(*pCurMap)[c].insert(Location(row, x));
			}
			(*positionOpp)[MX_Y - 1 - row][MX_X - 1 - x] = flip(c);
			++x;
		}
	}
	char arrposition[MX_Y][MX_X] = { 0 };
	char arrpositionOpp[MX_Y][MX_X] = { 0 };
	char(*position)[MX_Y][MX_X] = &arrposition;
	char(*positionOpp)[MX_Y][MX_X] = &arrpositionOpp;

	map<char, set<Location>> pieceMap;
	map<char, set<Location>> pieceMapOpp;

	map<char, set<Location>>* pCurMap;
	map<char, set<Location>>* pOppMap;

	string starter;
	string winner;
	int totalStep;
	string curSide;
	string oppSide;
	map<string, list<string>> moveMap;
};