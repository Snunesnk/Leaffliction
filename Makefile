CXX = gcc
CFLAGS = `pkg-config --cflags --libs opencv4`
LIB_UTILS = -lm -lstdc++
CXXFLAGS = -std=c++20 -Iinc  -Wall -Wextra

VPATH = src

PROGRAMS = distribution augmentation transformation
UTILS = $(VPATH)/image_processing.cpp $(VPATH)/image_utils.cpp
MODEL = train
MODEL_UTILS = $(VPATH)/model_calculate.cpp $(VPATH)/model_utils.cpp
OBJECTS = $(PROGRAMS:%=%.o)

.PHONY: all clean re

all: $(PROGRAMS) $(MODEL)


%: src/%.cpp src/image_processing.cpp src/image_utils.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CFLAGS) $(LIB_UTILS)

$(MODEL): $(VPATH)/$(MODEL).cpp $(UTILS) $(MODEL_UTILS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CFLAGS) $(LIB_UTILS)

clean:
	rm -rf $(PROGRAMS) $(MODEL)

re: clean all