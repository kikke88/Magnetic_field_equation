#include <mpi.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>
#include <functional>


#include <fftw3-mpi.h>


struct Task_features {
	const ptrdiff_t N;
	const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
	const double RANGE_LEFT, RANGE_RIGHT;
	const ptrdiff_t alloc_local, local_n0, local_0_start;
	double *indeces;
	int rank, size;

	Task_features(  const ptrdiff_t N_,
					const double rng_left_, const double rng_right_,
					const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_, const ptrdiff_t local_0_start_,
					const int rank_, const int size_):
					N{N_},
					INDEX_LEFT{-N / 2 + 1}, INDEX_RIGHT{N / 2 + 1}, //  [left, right)
					RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_},
					alloc_local{alloc_local_}, local_n0{local_n0_}, local_0_start{local_0_start_},
					rank{rank_}, size{size_} {
		indeces = new double[N];
		for (ptrdiff_t i = 0; i <= N / 2; ++i) {
			indeces[i] = i;
		}
		for (ptrdiff_t i = N / 2 + 1; i < N; ++i) {
			indeces[i] = i - N;
		}
	}

	~Task_features() {
		delete[] indeces;
	}
};


void derivative_of_function(fftw_complex* ptr, const Task_features& info, const int num_of_dimension) {
	double coef;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.INDEX_RIGHT; ++k) { //  [0, N/2 + 1)
				std::swap(	ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0],
							ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1]);
				if (num_of_dimension == 0) {
					coef = info.indeces[info.local_0_start + i];
				} else if (num_of_dimension == 1) {
					coef = info.indeces[j];
				} else if (num_of_dimension == 2) {
					coef = k;
				}
				ptr[(i * info.N + j) * (info.N / 2 + 1) + k][0] *= -coef;
				ptr[(i * info.N + j) * (info.N / 2 + 1) + k][1] *=  coef;
			}
		}
	}
	return;
}


void cross_product_func(double *cross_product,
						const double *v_f_l, const double *v_f_r,
						const double *m_f_l, const double *m_f_r,
						const Task_features& info) { // i/j/k * (v_f_l * m_f_l - v_f_r * m_f_r)
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.N; ++k) {
				const ptrdiff_t idx = (i * info.N + j) * (2 * (info.N / 2 + 1)) + k;
				cross_product[idx] = v_f_l[idx] * m_f_l[idx] - v_f_r[idx] * m_f_r[idx];
			}
		}
	}
	return;
}

void rotor(	fftw_complex *rot,
			const fftw_complex *cross_p_l, const fftw_complex *cross_p_r,
			const Task_features& info, const int num_of_dimension) {
	double coef_l, coef_r;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
				const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
				if (num_of_dimension == 0) {
					coef_l = info.indeces[j];
					coef_r = k;
				} else if (num_of_dimension == 1) {
					coef_l = k;
					coef_r = info.indeces[info.local_0_start + i];
				} else if (num_of_dimension == 2) {
					coef_l = info.indeces[info.local_0_start + i];
					coef_r = info.indeces[j];
				}
				rot[idx][0] = -cross_p_l[idx][1] * coef_l + cross_p_r[idx][1] * coef_r;
				rot[idx][1] =  cross_p_l[idx][0] * coef_l - cross_p_r[idx][0] * coef_r;
			}
		}
	}
	return;
}

void divergence(fftw_complex *result,
				const fftw_complex *ptr_1,
				const fftw_complex *ptr_2,
				const fftw_complex *ptr_3,
				const Task_features& info) {
	double coef_1, coef_2, coef_3;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
				const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
				coef_1 = info.indeces[info.local_0_start + i];
				coef_2 = info.indeces[j];
				coef_3 = k;
				result[idx][0] = -(coef_1 * ptr_1[idx][1] + coef_2 * ptr_2[idx][1] + coef_3 * ptr_3[idx][1]);
				result[idx][1] =   coef_1 * ptr_1[idx][0] + coef_2 * ptr_2[idx][0] + coef_3 * ptr_3[idx][0];
			}
		}
	}
	return;
}

double field_energy_phi(const double *ptr_1,
						const double *ptr_2,
						const double *ptr_3,
						const Task_features& info) {
	double energy = 0.;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.N; ++k) {
				const ptrdiff_t idx = (i * info.N + j) * (2 * (info.N / 2 + 1)) + k;
				energy += 	ptr_1[idx] * ptr_1[idx] +
							ptr_2[idx] * ptr_2[idx] +
							ptr_3[idx] * ptr_3[idx];
			}
		}
	}
	energy /= 2;
	return energy;
}

auto field_energy_fourie(	const fftw_complex *ptr_1,
							const fftw_complex *ptr_2,
							const fftw_complex *ptr_3,
							const Task_features& info) {
	double energy_real = 0., energy_imag = 0.;
	// for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
	// 	for (ptrdiff_t j = 0; j < info.N; ++j) {
	// 		ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1);
	// 		energy_real += 0.5 * (	ptr_1[idx][0] * ptr_1[idx][0] - ptr_1[idx][1] * ptr_1[idx][1] +
	// 								ptr_2[idx][0] * ptr_2[idx][0] - ptr_2[idx][1] * ptr_2[idx][1] +
	// 								ptr_3[idx][0] * ptr_3[idx][0] - ptr_3[idx][1] * ptr_3[idx][1]);
	// 		energy_imag += (ptr_1[idx][0] * ptr_1[idx][1] +
	// 						ptr_2[idx][0] * ptr_2[idx][1] +
	// 						ptr_3[idx][0] * ptr_3[idx][1]);
	// 		for (ptrdiff_t k = 1; k < info.RANGE_RIGHT; ++k) { //  [1, N/2 + 1)
	// 			++idx;
	// 			energy_real += (ptr_1[idx][0] * ptr_1[idx][0] - ptr_1[idx][1] * ptr_1[idx][1] +
	// 							ptr_2[idx][0] * ptr_2[idx][0] - ptr_2[idx][1] * ptr_2[idx][1] +
	// 							ptr_3[idx][0] * ptr_3[idx][0] - ptr_3[idx][1] * ptr_3[idx][1]);
	// 			energy_imag += 2 * (ptr_1[idx][0] * ptr_1[idx][1] +
	// 								ptr_2[idx][0] * ptr_2[idx][1] +
	// 								ptr_3[idx][0] * ptr_3[idx][1]);
	// 		}
	// 	}
	// }
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1);
			energy_real += 0.5 * (	ptr_1[idx][0] * ptr_1[idx][0] +
									ptr_2[idx][0] * ptr_2[idx][0] +
									ptr_3[idx][0] * ptr_3[idx][0] );
			energy_imag += (ptr_1[idx][0] * ptr_1[idx][1] +
							ptr_2[idx][0] * ptr_2[idx][1] +
							ptr_3[idx][0] * ptr_3[idx][1]);
			for (ptrdiff_t k = 1; k < info.RANGE_RIGHT; ++k) { //  [1, N/2 + 1)
				++idx;
				energy_real += (ptr_1[idx][0] * ptr_1[idx][0] +
								ptr_2[idx][0] * ptr_2[idx][0] +
								ptr_3[idx][0] * ptr_3[idx][0] );
				energy_imag += 2 * (ptr_1[idx][0] * ptr_1[idx][1] +
									ptr_2[idx][0] * ptr_2[idx][1] +
									ptr_3[idx][0] * ptr_3[idx][1]);
			}
		}
	}
	return std::make_pair(energy_real, energy_imag);
}

void correction(fftw_complex *result_ptr_1, fftw_complex *result_ptr_2, fftw_complex *result_ptr_3,
				fftw_complex *tmp_ptr, const Task_features& info) {
	divergence(tmp_ptr, result_ptr_1, result_ptr_2, result_ptr_3, info);
	double coef_1, coef_2, coef_3;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
				const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
				coef_1 = info.indeces[info.local_0_start + i];
				coef_2 = info.indeces[j];
				coef_3 = k;
				if (coef_1 == 0 and coef_2 == 0 and coef_3 == 0) {
					result_ptr_1[idx][0] = 0, result_ptr_1[idx][1] = 0,
					result_ptr_2[idx][0] = 0, result_ptr_2[idx][1] = 0,
					result_ptr_3[idx][0] = 0, result_ptr_3[idx][1] = 0;
				} else {
					double sum_coef = -(coef_1 * coef_1 + coef_2 * coef_2 + coef_3 * coef_3);
					result_ptr_1[idx][0] = result_ptr_1[idx][0] + coef_1 * tmp_ptr[idx][1] / sum_coef;
					result_ptr_1[idx][1] = result_ptr_1[idx][1] - coef_1 * tmp_ptr[idx][0] / sum_coef;
					result_ptr_2[idx][0] = result_ptr_2[idx][0] + coef_2 * tmp_ptr[idx][1] / sum_coef;
					result_ptr_2[idx][1] = result_ptr_1[idx][1] - coef_2 * tmp_ptr[idx][0] / sum_coef;
					result_ptr_3[idx][0] = result_ptr_3[idx][0] + coef_3 * tmp_ptr[idx][1] / sum_coef;
					result_ptr_3[idx][1] = result_ptr_1[idx][1] - coef_3 * tmp_ptr[idx][0] / sum_coef;
				}
			}
		}
	}
	return;
}

void output_of_large_abs_value_fourie_coef(	const fftw_complex *ptr,
											const Task_features& info,
											const double abs_value) {
	for (int rnk = 0; rnk < info.size; ++rnk) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (info.rank == rnk) {
			for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
				for (ptrdiff_t j = 0; j < info.N; ++j) {
					for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) {
						const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
						if (std::sqrt(ptr[idx][0] * ptr[idx][0] + ptr[idx][1] * ptr[idx][1]) > abs_value) {
							std::cout << "rank: " << rnk << " , idx: " << idx << " , value: " << ptr[idx][0] << ' ' << ptr[idx][1] << '\n';
						}
					}
				}
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	return;
}

void fill_real(double* vec[3], const Task_features& info) {
	std::vector<std::function<double(const ptrdiff_t, const ptrdiff_t, const ptrdiff_t)>> fill_real_functions;
	fill_real_functions.push_back([](const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t k) {
		return std::sin(0.3 * i - 1 * j + k);
	});
	fill_real_functions.push_back([](const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t k) {
		return std::cos(-1 * i - 0.1 * j - k);
	});
	fill_real_functions.push_back([](const ptrdiff_t i, const ptrdiff_t j, const ptrdiff_t k) {
		return std::sin(i + k);
	});

	for (ptrdiff_t q = 0; q < 3; ++q)
		for (ptrdiff_t i = 0; i < info.local_n0; ++i)
			for (ptrdiff_t j = 0; j < info.N; ++j)
				for (ptrdiff_t k = 0; k < info.N; ++k)
					vec[q][(i * info.N + j) * (2 * (info.N / 2 + 1)) + k] = fill_real_functions[q](i, j, k);


	return;
}


int main(int argc, char *argv[]) {
/*
	double 	*magnetic_field_1_real, *magnetic_field_2_real, *magnetic_field_3_real,
			*velocity_field_1_real, *velocity_field_2_real, *velocity_field_3_real,
			*cross_product_1_real, 	*cross_product_2_real, 	*cross_product_3_real;

	fftw_complex 	*magnetic_field_1_complex, 	*magnetic_field_2_complex, 	*magnetic_field_3_complex,
					*velocity_field_1_complex, 	*velocity_field_2_complex, 	*velocity_field_3_complex,
					*cross_product_1_complex,	*cross_product_2_complex, 	*cross_product_3_complex,
					*rotor_1_complex,			*rotor_2_complex,			*rotor_3_complex,
					*intermediate_field;
*/
//  Task1
	const int power_of_two = std::atoi(argv[1]);
	const ptrdiff_t N = 1 << power_of_two;

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	fftw_mpi_init();

	fftw_plan forward_plan[3], backward_plan[3];
	double* vec_r[3];
	fftw_complex* vec_c[3];
	ptrdiff_t alloc_local, local_n0, local_0_start;
	alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD, &local_n0, &local_0_start);

	const Task_features info{N, 0, std::acos(-1) * 2, alloc_local, local_n0, local_0_start, rank, size};

	for (int q = 0; q < 3; ++q) {
		vec_r[q] = fftw_alloc_real(2 * alloc_local);
		vec_c[q] = fftw_alloc_complex(alloc_local);
		forward_plan[q]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
		backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}

	fill_real(vec_r, info);

	std::cout << 'p' << '\n';
	std::cout << field_energy_phi(vec_r[0], vec_r[1], vec_r[2], info) << '\n';
	fftw_execute(forward_plan[0]);
	fftw_execute(forward_plan[1]);
	fftw_execute(forward_plan[2]);

	for (int q = 0; q < 3; ++q)
		for (ptrdiff_t i = 0; i < local_n0; ++i)
			for (ptrdiff_t j = 0; j < N; ++j)
				for (ptrdiff_t k = 0; k < (N / 2 + 1); ++k) {
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][0] /= N * std::sqrt(N);
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][1] /= N * std::sqrt(N);
				}

	const auto [real, imag] = field_energy_fourie(vec_c[0], vec_c[1], vec_c[2], info);
	std::cout << "f" << '\n';
	std::cout << real << ' ' << imag << '\n';
	// for (int q = 0; q < 3; ++q) {
	// 	std::cout << "##" << std::endl;
	// 	for (ptrdiff_t i = 0; i < local_n0; ++i)
	// 		for (ptrdiff_t j = 0; j < N; ++j)
	// 			for (ptrdiff_t k = 0; k < (N / 2 + 1); ++k)
	// 				std::cout << vec_c[q][(i * N + j) * (N / 2 + 1) + k][0] << ' ' << vec_c[q][(i * N + j) * (N / 2 + 1) + k][1] << std::endl;
	// }

	//derivative_of_function(cout, info, 0);

	// fftw_execute(backward_plan);
	// for (i = 0; i < local_n0; ++i)
	// 	for (j = 0; j < N; ++j)
	// 		for (k = 0; k < N; ++k)
	// 			rin[(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * sqrt(N);
	//
	// std::cout << field_energy_phi(rin, rin, rin, info) << '\n';

	for (int q = 0; q < 3; ++q) {
		fftw_free(vec_r[q]);
		fftw_free(vec_c[q]);
		fftw_destroy_plan(forward_plan[q]);
		fftw_destroy_plan(backward_plan[q]);
	}


	MPI_Finalize();

	return 0;
}
