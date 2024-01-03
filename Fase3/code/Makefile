SHELL = /bin/bash
BIN_NAME = MDpar

CXX = nvcc
LD  = nvcc

CXXFLAGS   = -O2 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC))
BIN = $(BIN_DIR)/$(BIN_NAME)

vpath %.cu $(SRC_DIR)

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	module load cuda/11.3.1;\
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BIN): $(OBJ)
	module load cuda/11.3.1;\
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ) $(LIBS)

checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN)

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/*

run: $(BIN)
	module load cuda/11.3.1;\
	sbatch test.sh