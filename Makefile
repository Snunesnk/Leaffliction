CXX = g++

CFLAGS = `pkg-config --cflags --libs opencv4`

CXXFLAGS = -std=c++20 -Iinc ${CFLAGS}  -Wall -Wextra

VPATH = src

PROGRAMS = distribution augmentation

OBJECTS = $(PROGRAMS:%=%.o)

.PHONY: all clean re

all: $(PROGRAMS)

%: src/%.cpp src/utils.cpp src/calculate.cpp src/image_processing.cpp src/image_utils.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(PROGRAMS:%=%.*)

re: clean all