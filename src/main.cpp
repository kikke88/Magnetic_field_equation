#include <mpi.h>
#include <cmath>
#include <iostream>
#include <utility>


#include <fftw3-mpi.h>

double b1_func( const double x1,
				const double x2,
				const double x3) {
	return std::exp(std::sin(x1 + 2 * x2 - x3));
}

double b2_func( const double x1,
				const double x2,
				const double x3) {
	return std::exp(std::cos(-8 * x1 + x2 - 9 * x3));
}

double b3_func( const double x1,
				const double x2,
				const double x3) {
	return std::exp(std::sin(x1 + -3 * x3 - 2 * x3));
}

struct Task_features {
	const ptrdiff_t N;
	const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
	const double RANGE_LEFT, RANGE_RIGHT;
	const ptrdiff_t alloc_local, local_n0, local_0_start;
	Task_features(  const ptrdiff_t N_,
					const double rng_left_, const double rng_right_,
					const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_, const ptrdiff_t local_0_start_):
					N{N_},
					INDEX_LEFT{-N / 2 + 1}, INDEX_RIGHT{N / 2 + 1}, //  [left, right)
					RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_},
					alloc_local{alloc_local_}, local_n0{local_n0_}, local_0_start{local_0_start_}
	{}
};

ptrdiff_t right_index(const ptrdiff_t idx, const ptrdiff_t N) {
	if (idx <= N / 2) {
		return idx;
	} else {
		return idx - N;
	}
}

void derivative_of_function(fftw_complex* ptr, const Task_features& info, const int num_of_dimension) {
	double coef;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.INDEX_RIGHT; ++k) { //  [0, N/2 + 1)
				std::swap(	ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0],
							ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1]);
				if (num_of_dimension == 0) {
					coef = right_index(info.local_0_start + i, info.N);
				} else if (num_of_dimension == 1) {
					coef = right_index(j, info.N);
				} else if (num_of_dimension == 2) {
					coef = info.INDEX_LEFT + k;
				} else {
					std::cerr << "AVOST" << '\n';
					exit(1);
				}
				std::cout << coef << '|' << '\n';
				ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0] *= -coef;
				ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1] *=  coef;
			}
		}
	}
	return;
}

fftw_complex divergence(fftw_complex *ptr_1, fftw_complex *ptr_2, fftw_complex *ptr_3, ptrdiff_t )////

int main(int argc, char *argv[]) {
	const int power_of_two = 1;
	const ptrdiff_t N = 1 << power_of_two;

	fftw_plan forward_plan, backward_plan;
	double *rin;
	fftw_complex *cout;
	ptrdiff_t alloc_local, local_n0, local_0_start, i, j, k;
	///
	double 	*magnetic_field_1_real, *magnetic_field_2_real, *magnetic_field_3_real,
			*velocity_field_1_real, *velocity_field_2_real, *velocity_field_3_real,
			*cross_product_1_real, 	*cross_product_2_real, 	*cross_product_2_real;

	fftw_complex 	*magnetic_field_1_complex, 	*magnetic_field_2_complex, 	*magnetic_field_3_complex,
					*velocity_field_1_complex, 	*velocity_field_2_complex, 	*velocity_field_3_complex,
					*cross_product_1_complex,	*cross_product_2_complex, 	*cross_product_2_complex,
					*rotor_1_complex, 			*rotor_2_complex, 			*rotor_3_complex,
					*intermediate_field;

	MPI_Init(&argc, &argv);
	fftw_mpi_init();
	//std::cout << -N / 2 << std::endl;
	/* get local data size and allocate */
	alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD,
										 &local_n0, &local_0_start);

	const Task_features info{N, 0, std::acos(-1) * 2, alloc_local, local_n0, local_0_start};

	rin = fftw_alloc_real(2 * alloc_local);
	cout = fftw_alloc_complex(alloc_local);

	/* create plan for out-of-place r2c DFT */
	forward_plan = fftw_mpi_plan_dft_r2c_3d(N, N, N, rin, cout, MPI_COMM_WORLD, FFTW_MEASURE);
	backward_plan = fftw_mpi_plan_dft_c2r_3d(N, N, N, cout, rin, MPI_COMM_WORLD, FFTW_MEASURE);

	/* initialize rin to some function my_func(x,y,z) */
	for (i = 0; i < local_n0; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				rin[(i * N + j) * (2 * (N / 2 + 1)) + k] = i + j + k;

	for (i = 0; i < local_n0; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				std::cout << rin[(i * N + j) * (2 * (N / 2 + 1)) + k] << std::endl;

	/* compute transforms as many times as desired */
	fftw_execute(forward_plan);
	//derivative_of_function(cout, info, 0);
	fftw_execute(backward_plan);



	// for (i = 0; i < local_n0; ++i)
	// 	for (j = 0; j < N; ++j)
	// 		for (k = 0; k < N; ++k)
	// 			std::cout << cout[(i * N + j) * (N / 2 + 1) + k][0] << ' ' << cout[(i * N + j) * (N / 2 + 1) + k][1] << std::endl;
	//derivative_of_function(cout, info, 0);

	std::cout << " --- " << '\n';
	// for (i = 0; i < local_n0; ++i)
	// 	for (j = 0; j < N; ++j)
	// 		for (k = 0; k < N; ++k)
	// 			std::cout << cout[(i * N + j) * (N / 2 + 1) + k][0] << ' ' << cout[(i * N + j) * (N / 2 + 1) + k][1] << std::endl;
	for (i = 0; i < local_n0; ++i)
		for (j = 0; j < N; ++j)
			for (k = 0; k < N; ++k)
				std::cout << rin[(i * N + j) * (2 * (N / 2 + 1)) + k] / 8 << std::endl;


	fftw_free(rin);
	fftw_free(cout);
	fftw_destroy_plan(forward_plan);
	fftw_destroy_plan(backward_plan);

	MPI_Finalize();

	return 0;
}
