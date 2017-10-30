#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <memory>

#include "Board.h"

using namespace std;

//
//string ExePath() {
//	char buffer[MAX_PATH];
//	//GetModuleFileName(NULL, buffer, MAX_PATH);
//	GetCurrentDirectory(MAX_PATH, buffer);
//	string::size_type pos = string(buffer).find_last_of("\\/");
//	return string(buffer).substr(0, pos);
//}
void splitLine(string line, vector<string>& retList) {
	stringstream ss(line);
	string data;
	while (getline(ss, data, ',')) {
		retList.push_back(data);
	}
}
void processGame(list< vector<string> >& game, ofstream& outFile, vector<shared_ptr<Position> >& result) {
	if (isdebug)
		cout << "processing game: " << game.front()[0] << endl;
	static int i = 0;
	Board board(game.front()[0]);

	for (auto data : game) {

		board.process(data);
	}

	board.finish();

	result.insert(result.end(), board.positionList.begin(), board.positionList.end());

	if (++i % 100 == 0) {
		cout << "processed 100 games" << endl;
		cout << "result size:" << result.size() << endl;
	}

	game.clear();
}

int main()
{
	//cout << ExePath() << endl;

	ifstream ifs;
	ifs.open("moves.csv");
	ofstream fp;
	fp.open("resultrandom.txt");
	string line;
	if (!getline(ifs, line))
	{
		cout << "empty file!" << endl;
		return -1;
	}
	cout << "header: " << line << endl;
	list< vector<string> > game;
	vector< shared_ptr<Position> > result;
	result.reserve(800000);
	while (getline(ifs, line)) {
		vector<string> data;
		splitLine(line, data);
		if (!game.empty() && game.back()[0] != data[0]) {
			//game is completed and can be processed
			processGame(game, fp, result);
		}
		game.push_back(data);
	}
	if (!game.empty())
		processGame(game, fp, result);

	cout << "going to shuffle" << endl;
	random_shuffle(result.begin(), result.end());
	for (auto p : result) {

		Position& pos = *p;
		if (pos.locLastFrom.valid()) {
			fp << pos.locLastFrom.x << " " << pos.locLastFrom.y << " " << pos.locLastTo.x << " " << pos.locLastTo.y << endl;
		}
		else {
			fp << endl;
		}
		fp << pos.locFrom.x << " " << pos.locFrom.y << " " << pos.locTo.x << " " << pos.locTo.y << endl;
		fp << pos.win << " " << pos.step << endl;
		for (int y = 0; y < MX_Y; ++y) {
			for (int x = 0; x < MX_X; ++x) {
				fp << pos.position[y][x];
			}
			fp << endl;
		}
	}
	cout << "Done" << endl;

    return 0;
}

