OPT 		=	-O2 -std=c++1z
FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm

all: task1

task1: bin/task1

bin/task1: build/task1.o Makefile
	mkdir -p bin
	mpic++ -o $@ $(OPT) $< $(FFTW_FLAGS)

build/task1.o: src/task1.cpp Makefile
	mkdir -p build
	mpic++ -o $@ $(OPT) -c $< $(FFTW_FLAGS)

clean:
	rm -rf build

clear:
	rm -rf bin lib
