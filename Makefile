CXX = g++
CFLAGS = `pkg-config --cflags --libs opencv4`
CXXFLAGS = -std=c++20 -Iinc ${CFLAGS}  -Wall -Wextra

VPATH = src

PROGRAMS = distribution augmentation
UTILS = $(VPATH)/image_processing.cpp $(VPATH)/image_utils.cpp
MODEL = train
MODEL_UTILS = $(VPATH)/model_calculate.cpp $(VPATH)/model_utils.cpp
OBJECTS = $(PROGRAMS:%=%.o)

.PHONY: all clean re

all: $(PROGRAMS) $(MODEL)

%: $(VPATH)/%.cpp $(UTILS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(MODEL): $(VPATH)/$(MODEL).cpp $(UTILS) $(MODEL_UTILS)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -rf $(PROGRAMS) $(MODEL)

re: clean all