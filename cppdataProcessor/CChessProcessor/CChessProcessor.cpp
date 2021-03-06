// CChessProcessor.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "Board.h"

using namespace std;


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
void processGame(list< vector<string> >& game, ofstream& outFile) {
	if (isdebug)
		cout << "processing game: " << game.front()[0] << endl;
	static int i = 0;
	Board board(outFile, game.front()[0]);

	for (auto data : game) {

		board.process(data);		
	}

	board.finish();
	if (++i % 100 == 0) {
		cout << "processed 100 games" << endl;
	}

	game.clear();
}

int main()
{
	cout << ExePath() << endl;

	ifstream ifs;
	ifs.open("moves.csv");
	ofstream ofs;
	ofs.open("result.txt");
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
		if (!game.empty() && game.back()[0] != data[0]) {
			//game is completed and can be processed
			processGame(game, ofs);			
		}
		game.push_back(data);
	}
	if (!game.empty()) 
		processGame(game, ofs);

	cout << "Done" << endl;

    return 0;
}

