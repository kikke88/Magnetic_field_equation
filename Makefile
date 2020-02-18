OPT 		=	-O2 -std=c++11
FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm
INC			=	-I ./include/

all: task1 task2

task1: bin/task1
task2: bin/task2

bin/task1: build/task1.o
	mkdir -p bin
	mpic++ -o $@ $(OPT) $< $(FFTW_FLAGS)

build/task1.o: src/task1.cpp
	mkdir -p build
	mpic++ -o $@ $(OPT) -c $< $(FFTW_FLAGS)


bin/task2: build/task2.o build/Field.o
	mkdir -p bin
	mpic++ -o $@ $(OPT) $(INC) $^ $(FFTW_FLAGS)

build/task2.o: src/task2.cpp
	mkdir -p build
	mpic++ -o $@ $(OPT) $(INC) -c $< $(FFTW_FLAGS)

build/Field.o: src/Field.cpp include/Field.hpp
	mkdir -p build
	mpic++ -o $@ $(OPT) $(INC) -c $< $(FFTW_FLAGS)

clean:
	rm -rf build

clear:
	rm -rf bin lib
