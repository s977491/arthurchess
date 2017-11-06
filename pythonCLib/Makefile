BOOST_ROOT = /media/sf_python/boost_1_65_1
CXX=/usr/bin/g++
CXXFLAGS =	-I/usr/include/python3.5m -std=c++11 -O3 -g -Wall -ansi -shared
OBJS =		spammodule.o 	

LIBS =

TARGET =	spammodule.so

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS) $(CXXFLAGS)

all:	clean	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
；