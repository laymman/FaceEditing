
#
# Makefile for preprocess.
#

CC  = g++
CXX = g++

INCLUDES = -I./face_preprocess/include
CFLAGS   = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)
CFLAGS += -std=c++11
CXXFLAGS += -std=c++11

LDFLAGS = -g -L./face_preprocess/lib/x64
LDLIBS = -lVIPLFaceDetector510 -lholiday -lVIPLFaceCrop500 -lVIPLPointDetector500

preprocess: face_preprocess/preprocess.cpp
	g++ -g -Wall -I./face_preprocess/include -std=c++11 -g \
	-L./face_preprocess/lib/x64 face_preprocess/preprocess.cpp \
	-lVIPLFaceDetector510 -lholiday -lVIPLFaceCrop500 -lVIPLPointDetector500 \
	-o preprocess `pkg-config --libs --cflags opencv` -ldl


.PHONY: clean
clean:
	rm -f *.o *~ a.out core preprocess

.PHONY: all
all: clean preprocess

