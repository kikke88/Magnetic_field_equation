#include <chrono>
#include <utility>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>

#include <mpi.h>
#include <fftw3-mpi.h>

#include <Field.hpp>

const double PI_PLUS_PI = std::acos(-1) * 2;


int main(int argc, char *argv[]) {

	const int iters     = std::atoi(argv[1]);
	const ptrdiff_t N   = std::atoi(argv[2]);
	const double    tau = std::strtod(argv[3], nullptr),
	                eta = std::strtod(argv[4], nullptr);
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	fftw_mpi_init();

	ptrdiff_t alloc_local, local_dim0_size, local_dim0_start;
	alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD, &local_dim0_size, &local_dim0_start);
	std::ofstream ofile("file_w_energy.data");
	{
		Field   magnetic_field  {Modes::ALL_COMPONENTS,         N, tau, eta, 0, PI_PLUS_PI, alloc_local, local_dim0_size, local_dim0_start, rank, size},
		        velocity_field  {Modes::ALL_REAL_COMPONENTS,    N, tau, eta, 0, PI_PLUS_PI, alloc_local, local_dim0_size, local_dim0_start, rank, size},
		        rotor_field     {Modes::ALL_COMPONENTS,         N, tau, eta, 0, PI_PLUS_PI, alloc_local, local_dim0_size, local_dim0_start, rank, size},
		        auxiliary_field {Modes::ONE_COMPLEX_COMPONENT,  N, tau, eta, 0, PI_PLUS_PI, alloc_local, local_dim0_size, local_dim0_start, rank, size};

		magnetic_field.fill_magnetic_field();
		velocity_field.fill_velocity_field();
		magnetic_field.forward_transformation();

		double cur_energy;
		for (int step = 0; step < iters; ++step) {
			magnetic_field.correction(auxiliary_field); // div = 0
			magnetic_field.backward_transformation();
			magnetic_field.do_step(velocity_field, rotor_field);
			cur_energy = magnetic_field.energy_fourie();
			if (cur_energy > 1e285) {
					break;
			}
			if (rank == 0) {
				std::cout << cur_energy << '\n';
				ofile << cur_energy << '\n';
			}
		}
	}
	ofile.close();
	MPI_Finalize();
	return 0;
}

/*
const auto begin = std::chrono::steady_clock::now();

const auto end = std::chrono::steady_clock::now();
auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
std::cout << elapsed_ms.count() << "ms" << "\n";
*/
