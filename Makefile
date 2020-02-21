FLAGS2 		=	-O2 -std=c++11 -Wall -Wextra -Wpedantic -Wshadow -Wfloat-equal -Wconversion -Wlogical-op -Wshift-overflow=2 -Wduplicated-cond -fsanitize=address -fsanitize=undefined
FLAGS1		=	-O2 -std=c++11 -Wall
FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm
INC			=	-I ./include/2/

all: task1 task2

task1: bin/task1
task2: bin/task2

bin/task1: build/task1.o
	mkdir -p bin
	mpic++ -o $@ $(FLAGS1) $< $(FFTW_FLAGS)

build/task1.o: src/1/task1.cpp
	mkdir -p build
	mpic++ -o $@ $(FLAGS1) -c $< $(FFTW_FLAGS)


bin/task2: build/task2.o build/Field.o
	mkdir -p bin
	mpic++ -o $@ $(FLAGS2) $(INC) $^ $(FFTW_FLAGS)

build/task2.o: src/2/task2.cpp
	mkdir -p build
	mpic++ -o $@ $(FLAGS2) $(INC) -c $< $(FFTW_FLAGS)

build/Field.o: src/2/Field.cpp include/2/Field.hpp
	mkdir -p build
	mpic++ -o $@ $(FLAGS2) $(INC) -c $< $(FFTW_FLAGS)

clean:
	rm -rf build

clear:
	rm -rf bin lib
