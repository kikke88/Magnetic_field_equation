# Magnetic_field_equation
[![S.png](https://i.postimg.cc/MHww7f8C/S.png)](https://postimg.cc/mzmKRDkj) 
Vector-functions: b - magnetic field, v - velocity field. η - float const.
b(0, x1, x2, x3), v(x1, x2, x3) - are known.
The problem is solved in the region [0, T] × Ω, where Ω = [0, 2π] × [0, 2π] × [0, 2π].
b(t, x1, x2, x3) = b(t, x1+2π, x2+2π, x3+2π)
To solve this system of equations, the three-dimensional Fourier transform and time integration of the obtained Fourier coefficients are used.

The main advantages of this method:

- The high order of approximation is O (1 / N ^ N), where N is the number of points on the segment.
- Convenient calculation of derivatives of any order
- On modern architectures, this method is well parallelized.

***

The program uses [FFTW library](http://www.fftw.org/). A distributed-memory MPI FFT implementation.

***

Task1 - testing of key functions(derivative of function, div, rot, energy estimation).
Task2 - solving the equation of change in the magnetic field.

***

Building

	make

Run

	mpirun -n *procs num* ./bin/task1 *num of points on the segment [0, 2π]*
	
or

	mpirun -n *procs num* ./bin/task2 	*num of time steps*
						*num of points on the segment [0, 2π]*
						*time step value*
						*η value*
	
The scheme is explicit, therefore, a time step cannot be taken large, a value of the order of 1e-4 should be sufficient. At first there will be a transitional mode, since the solution starts from some initial field, but starting from a certain moment, either exponential growth or exponential decay(depends on η) should be outlined.	

