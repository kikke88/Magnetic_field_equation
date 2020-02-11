OPT=			-O2 -std=c++1z
FFTW_FLAGS = 	-lfftw3_mpi -lfftw3 -lm

all: main

main: bin/main

bin/main: build/main.o Makefile
	mkdir -p bin
	mpic++ -o $@ $(OPT) $< $(FFTW_FLAGS)

build/main.o: src/main.cpp Makefile
	mkdir -p build
	mpic++ -o $@ $(OPT) -c $< $(FFTW_FLAGS)

clean:
	rm -rf build

clear:
	rm -rf bin lib
