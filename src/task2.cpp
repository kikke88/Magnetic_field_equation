#include <mpi.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>
#include <functional>
#include <chrono>

#include <fftw3-mpi.h>

const double EPSILON = 1e-10;

enum сlass  Modes {
	ALL_COMPONENTS,
	ALL_COMPLEX_COMPONENTS,
	ONE_COMPLEX_COMPONENT
};

//сделать отдельным файлом
class Field {
private:
	double* vec_r[3];
	fftw_complex* vec_c[3];
	fftw_plan forward_plan[3], backward_plan[3];

	const Modes mode;

	const ptrdiff_t N;
	const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
	const double RANGE_LEFT, RANGE_RIGHT;

	const ptrdiff_t alloc_local, local_n0, local_0_start;

	double *indeces;
	const double NORMALIZATION_CONSTANT;
	int rank, size;

public:
	Field(  const Mode mode_, const ptrdiff_t N_,
			const double rng_left_, const double rng_right_,
			const ptrdiff_t alloc_local_, const ptrdiff_t local_n0_, const ptrdiff_t local_0_start_,
			const int rank_, const int size_):
			mode(mode_), N{N_},
			INDEX_LEFT{-N / 2 + 1}, INDEX_RIGHT{N / 2 + 1}, //  [left, right)
			RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_},
			alloc_local{alloc_local_}, local_n0{local_n0_}, local_0_start{local_0_start_},
			normalization_constant{N * std::sqrt(N)}, rank{rank_}, size{size_} {
		if (mode == Modes::ALL_COMPONENTS) {
			for (int i = 0; i < 3; ++i) {
				vec_r[i] = fftw_alloc_real(2 * alloc_local);
				vec_c[i] = fftw_alloc_complex(alloc_local);
				forward_plan[i]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[i], vec_c[i], MPI_COMM_WORLD, FFTW_MEASURE);
				backward_plan[i] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[i], vec_r[i], MPI_COMM_WORLD, FFTW_MEASURE);
			}
		} else if (mode == Modes::ALL_COMPLEX_COMPONENTS) {
			for (int i = 0; i < 3; ++i) {
				vec_c[i] = fftw_alloc_complex(alloc_local);
			}
		} else if (mode == Modes::ONE_COMPLEX_COMPONENT){
			vec_c[0] = fftw_alloc_complex(alloc_local);
		}
		indeces = new double[N];
		for (ptrdiff_t i = 0; i <= N / 2; ++i) {
			indeces[i] = i;
		}
		for (ptrdiff_t i = N / 2 + 1; i < N; ++i) {
			indeces[i] = i - N;
		}
	}
	~Field() {
		delete[] indeces;
		if (mode == Modes::ALL_COMPONENTS) {
			for (int i = 0; i < 3; ++i) {
				delete[] vec_r[i];
				delete[] vec_c[i];
			}
		} else if (mode == Modes::ALL_COMPLEX_COMPONENTS) {
			for (int i = 0; i < 3; ++i) {
				delete[] vec_c[i];
			}
		} else if (mode == Modes::ONE_COMPLEX_COMPONENT){
			delete[] vec_c[0];
		}
	}

	void forward_transformation() {
		fftw_execute(forward_plan[0]);
		fftw_execute(forward_plan[1]);
		fftw_execute(forward_plan[2]);

		ptrdiff_t idx;
		for (ptrdiff_t i = 0; i < local_n0; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
					idx = (i * N + j) * (N / 2 + 1) + k;
					vec_c[0][idx][0] /= NORMALIZATION_CONSTANT;
					vec_c[0][idx][1] /= NORMALIZATION_CONSTANT;
					vec_c[1][idx][0] /= NORMALIZATION_CONSTANT;
					vec_c[1][idx][1] /= NORMALIZATION_CONSTANT;
					vec_c[2][idx][0] /= NORMALIZATION_CONSTANT;
					vec_c[2][idx][1] /= NORMALIZATION_CONSTANT;
				}
			}
		}
		return;
	}

	void backward_transformation() {
		fftw_execute(backward_plan[0]);
		fftw_execute(backward_plan[1]);
		fftw_execute(backward_plan[2]);

		ptrdiff_t idx;
		for (ptrdiff_t i = 0; i < local_n0; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < N; ++k) { //  [0, N/2 + 1)
					idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
					vec_r[0][idx] /= NORMALIZATION_CONSTANT;
					vec_r[1][idx] /= NORMALIZATION_CONSTANT;
					vec_r[2][idx] /= NORMALIZATION_CONSTANT;
				}
			}
		}
		return;
	}

	void divergence(const Field source_field) { // use only for ONE_COMPLEX_COMPONENT mode
		double coef_0, coef_0, coef_1;
		ptrdiff_t idx;
		for (ptrdiff_t i = 0; i < local_n0; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
					idx = (i * N + j) * (N / 2 + 1) + k;
					coef_0 = indeces[local_0_start + i];
					coef_0 = indeces[j];
					coef_1 = k;
					vec_c[0][idx][0] = -(   coef_0 * source_field.vec_c[0][idx][1] +
											coef_0 * source_field.vec_c[1][idx][1] +
											coef_1 * source_field.vec_c[2][idx][1]);
					vec_c[0][idx][1] = 	coef_0 * source_field.vec_c[0][idx][0] +
										coef_0 * source_field.vec_c[1][idx][0] +
										coef_1 * source_field.vec_c[2][idx][0];
				}
			}
		}
		return;
	}

	void correction(const Field tmp_field) {////////////дописать
		tmp_field.divergence(*this);
		/*

		проверка, что дивергенция равна 0

		*/
		double coef_0, coef_1, coef_2, sum_coef;
		ptrdiff_t idx;
		for (ptrdiff_t i = 0; i < local_n0; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
					idx = (i * N + j) * (N / 2 + 1) + k;
					coef_0 = indeces[local_0_start + i];
					coef_1 = indeces[j];
					coef_2 = k;
					if (coef_0 == 0 and coef_1 == 0 and coef_2 == 0) {
						vec_c[0][idx][0] = 0, vec_c[0][idx][1] = 0;
						vec_c[1][idx][0] = 0, vec_c[1][idx][1] = 0;
						vec_c[2][idx][0] = 0, vec_c[2][idx][1] = 0;
					} else {
						sum_coef = -(coef_0 * coef_0 + coef_1 * coef_1 + coef_2 * coef_2);
						vec_c[0][idx][0] = vec_c[0][idx][0] + coef_0 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[0][idx][1] = vec_c[0][idx][1] - coef_0 * tmp_field.vec_c[0][idx][0] / sum_coef;
						vec_c[1][idx][0] = vec_c[1][idx][0] + coef_1 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[1][idx][1] = vec_c[0][idx][1] - coef_1 * tmp_field.vec_c[0][idx][0] / sum_coef;
						vec_c[2][idx][0] = vec_c[2][idx][0] + coef_2 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[2][idx][1] = vec_c[2][idx][1] - coef_2 * tmp_field.vec_c[0][idx][0] / sum_coef;
					}
				}
			}
		}
		return;
	}

	void rotor(const Field velocity_field...../////////////////////)



};
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

/*
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
*/

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
	double coef_0, coef_1, coef_2;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
				const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
				coef_0 = info.indeces[info.local_0_start + i];
				coef_1 = info.indeces[j];
				coef_2 = k;
				result[idx][0] = -(coef_0 * ptr_1[idx][1] + coef_1 * ptr_2[idx][1] + coef_2 * ptr_3[idx][1]);
				result[idx][1] =   coef_0 * ptr_1[idx][0] + coef_1 * ptr_2[idx][0] + coef_2 * ptr_3[idx][0];
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

double field_energy_fourie(	const fftw_complex *ptr_1,
							const fftw_complex *ptr_2,
							const fftw_complex *ptr_3,
							const Task_features& info) {
	double energy = 0.;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1);
			energy += 0.5 * (	ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
								ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
								ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
			for (ptrdiff_t k = 1; k < info.RANGE_RIGHT; ++k) { //  [1, N/2 + 1)
				++idx;
				energy += (	ptr_1[idx][0] * ptr_1[idx][0] + ptr_1[idx][1] * ptr_1[idx][1] +
							ptr_2[idx][0] * ptr_2[idx][0] + ptr_2[idx][1] * ptr_2[idx][1] +
							ptr_3[idx][0] * ptr_3[idx][0] + ptr_3[idx][1] * ptr_3[idx][1]);
			}
		}
	}
	return energy;
}

void correction(fftw_complex *vec_c[0], fftw_complex *vec_c[1], fftw_complex *vec_c[2],
				fftw_complex *tmp_field.vec_c[0], const Task_features& info) {
	divergence(tmp_field.vec_c[0], vec_c[0], vec_c[1], vec_c[2], info);
	double coef_0, coef_1, coef_2;
	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) { //  [0, N/2 + 1)
				const ptrdiff_t idx = (i * info.N + j) * (info.N / 2 + 1) + k;
				coef_0 = info.indeces[info.local_0_start + i];
				coef_1 = info.indeces[j];
				coef_2 = k;
				if (coef_0 == 0 and coef_1 == 0 and coef_2 == 0) {
					vec_c[0][idx][0] = 0, vec_c[0][idx][1] = 0,
					vec_c[1][idx][0] = 0, vec_c[1][idx][1] = 0,
					vec_c[2][idx][0] = 0, vec_c[2][idx][1] = 0;
				} else {
					double sum_coef = -(coef_0 * coef_0 + coef_1 * coef_1 + coef_2 * coef_2);
					vec_c[0][idx][0] = vec_c[0][idx][0] + coef_0 * tmp_field.vec_c[0][idx][1] / sum_coef;
					vec_c[0][idx][1] = vec_c[0][idx][1] - coef_0 * tmp_field.vec_c[0][idx][0] / sum_coef;
					vec_c[1][idx][0] = vec_c[1][idx][0] + coef_1 * tmp_field.vec_c[0][idx][1] / sum_coef;
					vec_c[1][idx][1] = vec_c[0][idx][1] - coef_1 * tmp_field.vec_c[0][idx][0] / sum_coef;
					vec_c[2][idx][0] = vec_c[2][idx][0] + coef_2 * tmp_field.vec_c[0][idx][1] / sum_coef;
					vec_c[2][idx][1] = vec_c[0][idx][1] - coef_2 * tmp_field.vec_c[0][idx][0] / sum_coef;
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
	std::vector<std::function<double(const double, const double, const double)>> fill_real_functions;
	fill_real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::sin(x1 - 2 * x2 + 3 * x3);
	});
	fill_real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::cos(-x1 - x3);
	});
	fill_real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-3 * x1 - x2 + x3);
	});

	for (ptrdiff_t q = 0; q < 3; ++q)
		for (ptrdiff_t i = 0; i < info.local_n0; ++i)
			for (ptrdiff_t j = 0; j < info.N; ++j)
				for (ptrdiff_t k = 0; k < info.N; ++k) {
					const double cur_x = info.RANGE_RIGHT * (info.local_0_start + i) / info.N;
					const double cur_y = info.RANGE_RIGHT * j / info.N;
					const double cur_z = info.RANGE_RIGHT * k / info.N;
					vec[q][(i * info.N + j) * (2 * (info.N / 2 + 1)) + k] = fill_real_functions[q](cur_x, cur_y, cur_z);
				}


	return;
}

void test_derivative(const Task_features& info) {
	const ptrdiff_t N = info.N;
	fftw_plan forward_plan[3], backward_plan[3];
	double* vec_r[3];
	fftw_complex* vec_c[3];
	for (int q = 0; q < 3; ++q) {
		vec_r[q] = fftw_alloc_real(2 * info.alloc_local);
		vec_c[q] = fftw_alloc_complex(info.alloc_local);
		forward_plan[q]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
		backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}
	std::vector<std::vector<std::function<double(const double, const double, const double)>>> fill_real_derived_function(3);
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return -2 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return 3 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-x1 - x3);
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return 0;
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-x1 - x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return -3 * std::cos(-3 * x1 - x2 + x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return -std::cos(-3 * x1 - x2 + x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return std::cos(-3 * x1 - x2 + x3);
	});
	for (int num = 0; num < 3; ++num) {
		std::cout << num << '\n';
		fill_real(vec_r, info);
		fftw_execute(forward_plan[0]);
		fftw_execute(forward_plan[1]);
		fftw_execute(forward_plan[2]);

		derivative_of_function(vec_c[0], info, num);
		derivative_of_function(vec_c[1], info, num);
		derivative_of_function(vec_c[2], info, num);

		fftw_execute(backward_plan[0]);
		fftw_execute(backward_plan[1]);
		fftw_execute(backward_plan[2]);

		for (int q = 0; q < 3; ++q) {
			for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
				for (ptrdiff_t j = 0; j < info.N; ++j) {
					for (ptrdiff_t k = 0; k < info.N; ++k) {
						vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * N * N;
					}
				}
			}
		}

		for (int q = 0; q < 3; ++q) {
			for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
				for (ptrdiff_t j = 0; j < info.N; ++j) {
					for (ptrdiff_t k = 0; k < info.N; ++k) {
						const double cur_x = info.RANGE_RIGHT * (info.local_0_start + i) / info.N;
						const double cur_y = info.RANGE_RIGHT * j / info.N;
						const double cur_z = info.RANGE_RIGHT * k / info.N;
						const double diff = std::abs(vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] - fill_real_derived_function[q][num](cur_x, cur_y, cur_z));
						if (diff > EPSILON) {
							std::cout << info.rank << ' ' << i << ' ' << j << ' ' << k << std::endl;
						}
					}
				}
			}
		}
	}

	for (int q = 0; q < 3; ++q) {
		fftw_free(vec_r[q]);
		fftw_free(vec_c[q]);
		fftw_destroy_plan(forward_plan[q]);
		fftw_destroy_plan(backward_plan[q]);
	}

	return;
}

void test_rotor(const Task_features& info) {
	const ptrdiff_t N = info.N;
	fftw_plan forward_plan[3];
	double* vec_r[3];
	fftw_complex* vec_c[3];
	for (int q = 0; q < 3; ++q) {
		vec_r[q] = fftw_alloc_real(2 * info.alloc_local);
		vec_c[q] = fftw_alloc_complex(info.alloc_local);
		forward_plan[q]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}
	fill_real(vec_r, info);

	std::vector<std::function<double(const double, const double, const double)>> rotor_functions;
	rotor_functions.push_back([](const double x1, const double x2, const double x3) {
		return -std::cos(-3 * x1 - x2 + x3) - std::sin(-x1 - x3);
	});
	rotor_functions.push_back([](const double x1, const double x2, const double x3) {
		return 3 * std::cos(x1 - 2 * x2 + 3 * x3) + 3 * std::cos(-3 * x1 - x2 + x3);
	});
	rotor_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-x1 - x3) + 2 * std::cos(x1 - 2 * x2 + 3 * x3);
	});

	fftw_complex *rotor_c[3];
	double* rotor_r[3];
	fftw_plan rot_c_to_r[3];
	for (int q = 0; q < 3; ++q) {
		rotor_r[q] = fftw_alloc_real(2 * info.alloc_local);
		rotor_c[q] = fftw_alloc_complex(info.alloc_local);
		rot_c_to_r[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, rotor_c[q], rotor_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}

	fftw_execute(forward_plan[0]);
	fftw_execute(forward_plan[1]);
	fftw_execute(forward_plan[2]);

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < info.N; ++j) {
				for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) {
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][0] /= N * std::sqrt(N);
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][1] /= N * std::sqrt(N);
				}
			}
		}
	}

	rotor(rotor_c[0], vec_c[2], vec_c[1], info, 0);
	rotor(rotor_c[1], vec_c[0], vec_c[2], info, 1);
	rotor(rotor_c[2], vec_c[1], vec_c[0], info, 2);

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < info.N; ++j) {
				for (ptrdiff_t k = 0; k < info.N; ++k) {
					rotor_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * std::sqrt(N);
				}
			}
		}
	}

	fftw_execute(rot_c_to_r[0]);
	fftw_execute(rot_c_to_r[1]);
	fftw_execute(rot_c_to_r[2]);

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < info.N; ++j) {
				for (ptrdiff_t k = 0; k < info.N; ++k) {
					rotor_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * std::sqrt(N);
				}
			}
		}
	}

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < info.N; ++j) {
				for (ptrdiff_t k = 0; k < info.N; ++k) {
					const double cur_x = info.RANGE_RIGHT * (info.local_0_start + i) / info.N;
					const double cur_y = info.RANGE_RIGHT * j / info.N;
					const double cur_z = info.RANGE_RIGHT * k / info.N;
					const double diff = std::abs(rotor_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] - rotor_functions[q](cur_x, cur_y, cur_z));
					if (diff > EPSILON) {
						std::cout << info.rank << ' ' << i << ' ' << j << ' ' << k << std::endl;
					}
				}
			}
		}
	}

	for (int q = 0; q < 3; ++q) {
		fftw_free(vec_r[q]);
		fftw_free(vec_c[q]);
		fftw_free(rotor_r[q]);
		fftw_free(rotor_c[q]);
		fftw_destroy_plan(rot_c_to_r[q]);
		fftw_destroy_plan(forward_plan[q]);
	}

	return;
}

void test_divergence(const Task_features& info) {
	const ptrdiff_t N = info.N;
	fftw_plan forward_plan[3], backward_plan[3];
	double* vec_r[3];
	fftw_complex* vec_c[3];
	for (int q = 0; q < 3; ++q) {
		vec_r[q] = fftw_alloc_real(2 * info.alloc_local);
		vec_c[q] = fftw_alloc_complex(info.alloc_local);
		forward_plan[q]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
		backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}

	std::vector<std::vector<std::function<double(const double, const double, const double)>>> fill_real_derived_function(3); // i - func num, j - var num
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return -2 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[0].push_back([](const double x1, const double x2, const double x3) {
		return 3 * std::cos(1 * x1 - 2 * x2 + 3 * x3);
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-x1 - x3);
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return 0;
	});
	fill_real_derived_function[1].push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-x1 - x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return -3 * std::cos(-3 * x1 - x2 + x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return -std::cos(-3 * x1 - x2 + x3);
	});
	fill_real_derived_function[2].push_back([](const double x1, const double x2, const double x3) {
		return std::cos(-3 * x1 - x2 + x3);
	});

	fill_real(vec_r, info);

	fftw_execute(forward_plan[0]);
	fftw_execute(forward_plan[1]);
	fftw_execute(forward_plan[2]);

	derivative_of_function(vec_c[0], info, 0);
	derivative_of_function(vec_c[1], info, 1);
	derivative_of_function(vec_c[2], info, 2);

	fftw_execute(backward_plan[0]);
	fftw_execute(backward_plan[1]);
	fftw_execute(backward_plan[2]);

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < info.N; ++j) {
				for (ptrdiff_t k = 0; k < info.N; ++k) {
					vec_r[q][(i * N + j) * (2 * (N / 2 + 1)) + k] /= N * N * N;
				}
			}
		}
	}

	for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
		for (ptrdiff_t j = 0; j < info.N; ++j) {
			for (ptrdiff_t k = 0; k < info.N; ++k) {
				const double cur_x = info.RANGE_RIGHT * (info.local_0_start + i) / info.N;
				const double cur_y = info.RANGE_RIGHT * j / info.N;
				const double cur_z = info.RANGE_RIGHT * k / info.N;
				const double diff = std::abs(
					vec_r[0][(i * N + j) * (2 * (N / 2 + 1)) + k] +
					vec_r[1][(i * N + j) * (2 * (N / 2 + 1)) + k] +
					vec_r[2][(i * N + j) * (2 * (N / 2 + 1)) + k] -
					(	fill_real_derived_function[0][0](cur_x, cur_y, cur_z) +
						fill_real_derived_function[1][1](cur_x, cur_y, cur_z) +
						fill_real_derived_function[2][2](cur_x, cur_y, cur_z)));
				if (diff > EPSILON) {
					std::cout << info.rank << ' ' << i << ' ' << j << ' ' << k << std::endl;
				}
			}
		}
	}

	for (int q = 0; q < 3; ++q) {
		fftw_free(vec_r[q]);
		fftw_free(vec_c[q]);
		fftw_destroy_plan(forward_plan[q]);
		fftw_destroy_plan(backward_plan[q]);
	}

	return;
}

void test_energy(const Task_features& info) {
	const ptrdiff_t N = info.N;
	fftw_plan forward_plan[3], backward_plan[3];
	double* vec_r[3];
	fftw_complex* vec_c[3];
	for (int q = 0; q < 3; ++q) {
		vec_r[q] = fftw_alloc_real(2 * info.alloc_local);
		vec_c[q] = fftw_alloc_complex(info.alloc_local);
		forward_plan[q]  = fftw_mpi_plan_dft_r2c_3d(N, N, N, vec_r[q], vec_c[q], MPI_COMM_WORLD, FFTW_MEASURE);
		backward_plan[q] = fftw_mpi_plan_dft_c2r_3d(N, N, N, vec_c[q], vec_r[q], MPI_COMM_WORLD, FFTW_MEASURE);
	}
	fill_real(vec_r, info);

	double energy = field_energy_phi(vec_r[0], vec_r[1], vec_r[2], info);
	if (info.rank == 0) {
		MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	} else {
		MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	if (info.rank == 0) {
		std::cout << 'p' << '\n';
		std::cout << energy << '\n';
	}

	fftw_execute(forward_plan[0]);
	fftw_execute(forward_plan[1]);
	fftw_execute(forward_plan[2]);

	for (int q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < info.local_n0; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < info.RANGE_RIGHT; ++k) {
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][0] /= N * std::sqrt(N);
					vec_c[q][(i * N + j) * (N / 2 + 1) + k][1] /= N * std::sqrt(N);
				}
			}
		}
	}

	energy = field_energy_fourie(vec_c[0], vec_c[1], vec_c[2], info);
	if (info.rank == 0) {
		MPI_Reduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	} else {
		MPI_Reduce(&energy, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	if (info.rank == 0) {
		std::cout << 'f' << '\n';
		std::cout << energy << '\n';
	}

	for (int q = 0; q < 3; ++q) {
		fftw_free(vec_r[q]);
		fftw_free(vec_c[q]);
		fftw_destroy_plan(forward_plan[q]);
		fftw_destroy_plan(backward_plan[q]);
	}

	return;
}

/*
const auto begin = std::chrono::steady_clock::now();

const auto end = std::chrono::steady_clock::now();
auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
std::cout << elapsed_ms.count() << "ms" << "\n";
*/

int main(int argc, char *argv[]) {

	const int power_of_two = std::atoi(argv[1]);
	const ptrdiff_t N = 1 << power_of_two;

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	fftw_mpi_init();

	ptrdiff_t alloc_local, local_n0, local_0_start;
	alloc_local = fftw_mpi_local_size_3d(N, N, N / 2 + 1, MPI_COMM_WORLD, &local_n0, &local_0_start);

	const Task_features info{N, 0, std::acos(-1) * 2, alloc_local, local_n0, local_0_start, rank, size};

	//test_derivative(info);
	//test_rotor(info);
	//test_divergence(info);
	//test_energy(info);

	MPI_Finalize();
	return 0;
}
