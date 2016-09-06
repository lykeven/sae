# A single output portable Makefile for
# simple c++ project

OBJ_DIR = obj
BIN_DIR = bin
TARGET = sae

BIN_TARGET = $(BIN_DIR)/$(TARGET)

DEFINES +=

DIRNAME=$(shell basename `pwd`)


PKGCONFIG_LIBS =
INCLUDE_DIR = -Isrc/lib 

# CXXFLAGS += -O2
CXXFLAGS += -g -O0
CXXFLAGS += #-static -static-libstdc++
CXXFLAGS += -lpthread #-fopenmp  -lboost
CXXFLAGS += -lboost_thread-mt
CXXFLAGS += $(DEFINES)
CXXFLAGS += -std=c++11
CXXFLAGS += -Wall -Wextra
CXXFLAGS += $(INCLUDE_DIR)
CXXFLAGS += $(ARPACK_DIR)
# CXXFLAGS += $(shell pkg-config --cflags --libs $(PKGCONFIG_LIBS))
CXXFLAGS += #-m64

LDFLAGS =

CXX ?= g++
CC = gcc

CXXSOURCES = $(shell find -L src -name "*.cpp" -o -name '*.c')
OBJS = $(addsuffix .o, $(addprefix $(OBJ_DIR)/,$(CXXSOURCES)))
DEPFILES = $(OBJS:.o=.d)

.PHONY: all clean run rebuild gdb

all: $(BIN_TARGET)
	@echo "compilation succeed, issue 'make run' to execute the program."

$(OBJ_DIR)/%.c.o: %
	@echo "[cxx] $< ..."
	@$(CC) -c $< $(CXXFLAGS) -o $@

$(OBJ_DIR)/%.o: %
	@echo "[cxx] $< ..."
	@$(CXX) -c $< $(CXXFLAGS) -o $@


$(OBJ_DIR)/%.d: %
	@mkdir -p $(dir $@)
	@echo "[dep] $< ..."
	@$(CXX) $(INCLUDE_DIR) $(CXXFLAGS) -MM -MT "$(OBJ_DIR)/$(<:.cpp=.o) $(OBJ_DIR)/$(<:.cpp=.d)" "$<" > "$@"

sinclude $(DEPFILES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_TARGET): $(BIN_DIR) $(OBJS)
	@echo "[link] $< ..."
	@mkdir -p $(BIN_DIR)
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

rebuild:
	+@make clean
	+@make

run: $(BIN_TARGET)
	./$(BIN_TARGET) $(ARG)

gdb: $(BIN_TARGET)
	gdb ./$(BIN_TARGET)

