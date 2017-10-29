// CChessProcessor.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

using namespace std;

#define SIDE_RED "red"
#define SIDE_BLACK "black"
#define MX_X 9
#define MX_Y 10

class Location {
public:
	int x;
	int y;
	Location(int _x, int _y) : x(_x), y(_y) {}
};
class Board {
public:
	Board() {	
		resetBoard();
	}
	virtual ~Board() {
	}
	string toString() {
		//print the board, previous move, and target move
	}
	void resetBoard() {

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
	}
	void printBoard() {
		cout << "board now:" << endl;
		for (int y = 0; y < MX_Y; ++y) {
			for (int x = 0; x < MX_X; ++x) {
				cout << (*position)[y][x];
			}
			cout << endl;
		}
	}
	void process(const vector<string>& data) {
		if (starter.empty()) {
			starter = data[2];
			curSide = starter;
			if (curSide == SIDE_RED)
				oppSide = SIDE_BLACK;
			else
				oppSide = SIDE_RED;
		}
		moveMap[data[2]].push_back(data[3]);
	}
	void process(const string& move) {
		printBoard();
		cout << "going to move" << move << endl;
		flipSide();
	}
	void finish(const bool debug) {
		auto& blackMoveList = moveMap[SIDE_BLACK];
		auto& redMoveList = moveMap[SIDE_RED];
		auto pList = &redMoveList;
		auto pOppList = &blackMoveList;
		if (starter == SIDE_BLACK)
			swap(pList, pOppList);
		do {
			if (pList->empty())
				break;
			string move = pList->front();
			process(move);
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
			(*positionOpp)[MX_Y-1 - row][MX_X-1 - x] = flip(c);
			++x;
		}
	}
	char arrposition[MX_Y][MX_X] = { 0 };
	char arrpositionOpp[MX_Y][MX_X] = { 0 };
	char (* position)[MX_Y][MX_X]  = &arrposition;
	char(*positionOpp)[MX_Y][MX_X] = &arrpositionOpp;

	string starter;
	string curSide;
	string oppSide;
	map<string, list<string>> moveMap;
};
string ExePath() {
	char buffer[MAX_PATH];
	//GetModuleFileName(NULL, buffer, MAX_PATH);
	GetCurrentDirectory(MAX_PATH, buffer);
	string::size_type pos = string(buffer).find_last_of("\\/");
	return string(buffer).substr(0, pos);
}
void splitLine(string line, vector<string>& retList) {
	stringstream ss(line);
	string data;
	while (getline(ss, data, ',')) {
		retList.push_back(data);
	}
}
void processGame(list< vector<string> >& game) {
	cout << "processing game: " << game.front()[0] << endl;
	Board board;
	for (auto data : game) {
		board.process(data);
	}
	board.finish(true);

	game.clear();
}

int main()
{
	cout << ExePath() << endl;

	ifstream ifs;
	ifs.open("moves.csv");
	string line;
	if (!getline(ifs, line))
	{
		cout << "empty file!" << endl;
		return -1;
	}
	cout << "header: " << line << endl;
	list< vector<string> > game;
	while (getline(ifs, line)) {
		vector<string> data;
		splitLine(line, data);
		if (game.empty() || game.back()[0] == data[0]) {
			game.push_back(data);
		}
		else {
			//game is completed and can be processed
			processGame(game);
		}
	}
	if (!game.empty()) 
		processGame(game);

	cout << "Done" << endl;

    return 0;
}

