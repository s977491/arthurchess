CXXFLAGS =	-O2 -g -Wall  -std=c++11 -fmessage-length=0

OBJS =		Board.o DataProcessor.o

LIBS =

TARGET =	processor

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

all:	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
