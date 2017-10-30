#include "stdafx.h"
#include "Board.h"


std::ostream &operator<<(std::ostream &os, Location const &m) {
	return os << "LocXY:" << m.x << "," << m.y;
}