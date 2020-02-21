#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>

#include <mpi.h>
#include <fftw3-mpi.h>

#include <Field.hpp>


void Field::cross_product(const Field& velocity_field, const Field& magnetic_field) {
	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < N; ++k) {
				idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
				vec_r[0][idx] = velocity_field.vec_r[1][idx] * magnetic_field.vec_r[2][idx] -
								velocity_field.vec_r[2][idx] * magnetic_field.vec_r[1][idx];

				vec_r[1][idx] = velocity_field.vec_r[2][idx] * magnetic_field.vec_r[0][idx] -
								velocity_field.vec_r[0][idx] * magnetic_field.vec_r[2][idx];

				vec_r[2][idx] = velocity_field.vec_r[0][idx] * magnetic_field.vec_r[1][idx] -
								velocity_field.vec_r[1][idx] * magnetic_field.vec_r[0][idx];
			}
		}
	}

	return;
}

void Field::divergence(const Field& source_field) {
	double coef_0, coef_1, coef_2;
	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
				idx = (i * N + j) * (N / 2 + 1) + k;
				coef_0 = static_cast<double>(indeces[local_dim0_start + i]);
				coef_1 = static_cast<double>(indeces[j]);
				coef_2 = static_cast<double>(k);
				vec_c[0][idx][0] =-(coef_0 * source_field.vec_c[0][idx][1] +
									coef_1 * source_field.vec_c[1][idx][1] +
									coef_2 * source_field.vec_c[2][idx][1]);

				vec_c[0][idx][1] = 	coef_0 * source_field.vec_c[0][idx][0] +
									coef_1 * source_field.vec_c[1][idx][0] +
									coef_2 * source_field.vec_c[2][idx][0];
			}
		}
	}

	return;
}

Field::Field(const Modes mode_, const ptrdiff_t N_, const double TAU_, const double ETA_,
		const double rng_left_, const double rng_right_,
		const ptrdiff_t alloc_local_, const ptrdiff_t local_dim0_size_, const ptrdiff_t local_dim0_start_,
		const int rank_, const int size_):
		mode{mode_}, N{N_}, TAU{TAU_}, ETA{ETA_},
		INDEX_LEFT{-N / 2 + 1}, INDEX_RIGHT{N / 2 + 1}, // [,)
		RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_},
		alloc_local{alloc_local_}, local_dim0_size{local_dim0_size_}, local_dim0_start{local_dim0_start_},
		NORMALIZATION_CONSTANT{std::sqrt(N * N * N)}, rank{rank_}, size{size_} {
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
	} else if (mode == Modes::ALL_REAL_COMPONENTS) {
		for (int i = 0; i < 3; ++i) {
			vec_r[i] = fftw_alloc_real(2 * alloc_local);
		}
	} else if (mode == Modes::ONE_COMPLEX_COMPONENT) {
		vec_c[0] = fftw_alloc_complex(alloc_local);
	}

	indeces = new ptrdiff_t[N];
	for (ptrdiff_t i = 0; i <= N / 2; ++i) {
		indeces[i] = i;
	}
	for (ptrdiff_t i = N / 2 + 1; i < N; ++i) {
		indeces[i] = i - N;
	}
}

Field::~Field() {
	delete[] indeces;
	if (mode == Modes::ALL_COMPONENTS) {
		for (int i = 0; i < 3; ++i) {
			fftw_free(vec_r[i]);
			fftw_free(vec_c[i]);
			fftw_destroy_plan(forward_plan[i]);
			fftw_destroy_plan(backward_plan[i]);
		}
	} else if (mode == Modes::ALL_COMPLEX_COMPONENTS) {
		for (int i = 0; i < 3; ++i) {
			fftw_free(vec_c[i]);
		}
	} else if (mode == Modes::ALL_REAL_COMPONENTS) {
		for (int i = 0; i < 3; ++i) {
			fftw_free(vec_r[i]);
		}
	} else if (mode == Modes::ONE_COMPLEX_COMPONENT){
		fftw_free(vec_c[0]);
	}
}

void Field::forward_transformation() {
	fftw_execute(forward_plan[0]);
	fftw_execute(forward_plan[1]);
	fftw_execute(forward_plan[2]);

	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
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

void Field::backward_transformation() {
	fftw_execute(backward_plan[0]);
	fftw_execute(backward_plan[1]);
	fftw_execute(backward_plan[2]);

	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < N; ++k) {
				idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
				vec_r[0][idx] /= NORMALIZATION_CONSTANT;
				vec_r[1][idx] /= NORMALIZATION_CONSTANT;
				vec_r[2][idx] /= NORMALIZATION_CONSTANT;
			}
		}
	}

	return;
}

void Field::correction(Field& tmp_field) {
	tmp_field.divergence(*this);

	ptrdiff_t idx;
	double local_max = 0.;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
				idx = (i * N + j) * (N / 2 + 1) + k;
				local_max = std::max(local_max, std::sqrt(
												tmp_field.vec_c[0][idx][0] *
												tmp_field.vec_c[0][idx][0] +
												tmp_field.vec_c[0][idx][1] *
												tmp_field.vec_c[0][idx][1]));
			}
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &local_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

	if (local_max >= EPSILON) {
		double coef_0, coef_1, coef_2, sum_coef;
		for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
					idx = (i * N + j) * (N / 2 + 1) + k;
					coef_0 = static_cast<double>(indeces[local_dim0_start + i]);
					coef_1 = static_cast<double>(indeces[j]);
					coef_2 = static_cast<double>(k);
					if (std::abs(coef_0) + std::abs(coef_1) + std::abs(coef_2) < EPSILON) {
						vec_c[0][idx][0] = 0;
						vec_c[0][idx][1] = 0;
						vec_c[1][idx][0] = 0;
						vec_c[1][idx][1] = 0;
						vec_c[2][idx][0] = 0;
						vec_c[2][idx][1] = 0;
					} else {
						sum_coef = coef_0 * coef_0 + coef_1 * coef_1 + coef_2 * coef_2;	
						vec_c[0][idx][0] += -coef_0 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[0][idx][1] +=  coef_0 * tmp_field.vec_c[0][idx][0] / sum_coef;
						vec_c[1][idx][0] += -coef_1 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[1][idx][1] +=  coef_1 * tmp_field.vec_c[0][idx][0] / sum_coef;
						vec_c[2][idx][0] += -coef_2 * tmp_field.vec_c[0][idx][1] / sum_coef;
						vec_c[2][idx][1] +=  coef_2 * tmp_field.vec_c[0][idx][0] / sum_coef;
					}
				}
			}
		}
	}

	return;
}

void Field::rotor(const Field& velocity_field, const Field& magnetic_field) {
	cross_product(velocity_field, magnetic_field);

	forward_transformation();

	double coef_0_l, coef_0_r, coef_1_l, coef_1_r, coef_2_l, coef_2_r;
	double tmp_0_r, tmp_0_i, tmp_1_r, tmp_1_i, tmp_2_r, tmp_2_i;
	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
				idx = (i * N + j) * (N / 2 + 1) + k;
				coef_0_l = static_cast<double>(indeces[j]);
				coef_0_r = static_cast<double>(k);
				coef_1_l = static_cast<double>(k);
				coef_1_r = static_cast<double>(indeces[local_dim0_start + i]);
				coef_2_l = static_cast<double>(indeces[local_dim0_start + i]);
				coef_2_r = static_cast<double>(indeces[j]);
				tmp_0_r = vec_c[0][idx][0];
				tmp_0_i = vec_c[0][idx][1];
				tmp_1_r = vec_c[1][idx][0];
				tmp_1_i = vec_c[1][idx][1];
				tmp_2_r = vec_c[2][idx][0];
				tmp_2_i = vec_c[2][idx][1];
				vec_c[0][idx][0] = -tmp_2_i * coef_0_l + tmp_1_i * coef_0_r;
				vec_c[0][idx][1] =  tmp_2_r * coef_0_l - tmp_1_r * coef_0_r;
				vec_c[1][idx][0] = -tmp_0_i * coef_1_l + tmp_2_i * coef_1_r;
				vec_c[1][idx][1] =  tmp_0_r * coef_1_l - tmp_2_r * coef_1_r;
				vec_c[2][idx][0] = -tmp_1_i * coef_2_l + tmp_0_i * coef_2_r;
				vec_c[2][idx][1] =  tmp_1_r * coef_2_l - tmp_0_r * coef_2_r;
			}
		}
	}

	return;
}

void Field::do_step(const Field& velocity_field, Field& rotor_field) {

	rotor_field.rotor(velocity_field, *this);

	ptrdiff_t idx;
	double coef_0, coef_1, coef_2, sum_coef;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < INDEX_RIGHT; ++k) {
				idx = (i * N + j) * (N / 2 + 1) + k;
				coef_0 = static_cast<double>(indeces[local_dim0_start + i]);
				coef_1 = static_cast<double>(indeces[j]);
				coef_2 = static_cast<double>(k);
				sum_coef = coef_0 * coef_0 + coef_1 * coef_1 + coef_2 * coef_2;
				vec_c[0][idx][0] += (-ETA * sum_coef * vec_c[0][idx][0] + rotor_field.vec_c[0][idx][0]) * TAU;
				vec_c[0][idx][1] += (-ETA * sum_coef * vec_c[0][idx][1] + rotor_field.vec_c[0][idx][1]) * TAU;
				vec_c[1][idx][0] += (-ETA * sum_coef * vec_c[1][idx][0] + rotor_field.vec_c[1][idx][0]) * TAU;
				vec_c[1][idx][1] += (-ETA * sum_coef * vec_c[1][idx][1] + rotor_field.vec_c[1][idx][1]) * TAU;
				vec_c[2][idx][0] += (-ETA * sum_coef * vec_c[2][idx][0] + rotor_field.vec_c[2][idx][0]) * TAU;
				vec_c[2][idx][1] += (-ETA * sum_coef * vec_c[2][idx][1] + rotor_field.vec_c[2][idx][1]) * TAU;
			}
		}
	}

	return;
}

double Field::energy_phi() const {
	double energy = 0.;
	ptrdiff_t idx;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			for (ptrdiff_t k = 0; k < N; ++k) {
				idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
				energy += 	vec_r[0][idx] * vec_r[0][idx] +
							vec_r[1][idx] * vec_r[1][idx] +
							vec_r[2][idx] * vec_r[2][idx];
			}
		}
	}
	energy /= 2;

	MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return energy;
}

double Field::energy_fourie() const {
	double energy = 0.;
	for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
		for (ptrdiff_t j = 0; j < N; ++j) {
			ptrdiff_t idx = (i * N + j) * (N / 2 + 1);
			energy += 0.5 * (	vec_c[0][idx][0] * vec_c[0][idx][0] + vec_c[0][idx][1] * vec_c[0][idx][1] +
								vec_c[1][idx][0] * vec_c[1][idx][0] + vec_c[1][idx][1] * vec_c[1][idx][1] +
								vec_c[2][idx][0] * vec_c[2][idx][0] + vec_c[2][idx][1] * vec_c[2][idx][1]);
			for (ptrdiff_t k = 1; k < INDEX_RIGHT; ++k) {
				++idx;
				energy += (	vec_c[0][idx][0] * vec_c[0][idx][0] + vec_c[0][idx][1] * vec_c[0][idx][1] +
							vec_c[1][idx][0] * vec_c[1][idx][0] + vec_c[1][idx][1] * vec_c[1][idx][1] +
							vec_c[2][idx][0] * vec_c[2][idx][0] + vec_c[2][idx][1] * vec_c[2][idx][1]);
			}
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return energy;
}

void Field::fill_velocity_field() {
	const double tmp_const = 2 / std::sqrt(3);
	std::vector<std::function<double(const double, const double, const double)>> velocity_functions;
	velocity_functions.push_back([=](const double, const double x2, const double x3) {
		return tmp_const * std::sin(x2) * std::cos(x3);
	});
	velocity_functions.push_back([=](const double x1, const double, const double x3) {
		return tmp_const * std::sin(x3) * std::cos(x1);
	});
	velocity_functions.push_back([=](const double x1, const double x2, const double) {
		return tmp_const * std::sin(x1) * std::cos(x2);
	});

	double cur_x, cur_y, cur_z;
	ptrdiff_t idx;
	for (ptrdiff_t q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < N; ++k) {
					idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
					cur_x = RANGE_RIGHT * static_cast<double>(local_dim0_start + i) / static_cast<double>(N);
					cur_y = RANGE_RIGHT * static_cast<double>(j) / static_cast<double>(N);
					cur_z = RANGE_RIGHT * static_cast<double>(k) / static_cast<double>(N);
					vec_r[q][idx] = velocity_functions[q](cur_x, cur_y, cur_z);
				}
			}
		}
	}

	return;
}

void Field::fill_magnetic_field() {
	std::vector<std::function<double(const double, const double, const double)>> real_functions;
	real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::sin(x1 - 2 * x2 + 3 * x3);
	});
	real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::cos(-x1 - x3 + 5 * x2);
	});
	real_functions.push_back([](const double x1, const double x2, const double x3) {
		return std::sin(-3 * x1 - x2 + x3);
	});

	double cur_x, cur_y, cur_z;
	ptrdiff_t idx;
	for (ptrdiff_t q = 0; q < 3; ++q) {
		for (ptrdiff_t i = 0; i < local_dim0_size; ++i) {
			for (ptrdiff_t j = 0; j < N; ++j) {
				for (ptrdiff_t k = 0; k < N; ++k) {
					idx = (i * N + j) * (2 * (N / 2 + 1)) + k;
					cur_x = RANGE_RIGHT * static_cast<double>(local_dim0_start + i) / static_cast<double>(N);
					cur_y = RANGE_RIGHT * static_cast<double>(j) / static_cast<double>(N);
					cur_z = RANGE_RIGHT * static_cast<double>(k) / static_cast<double>(N);
					vec_r[q][idx] = real_functions[q](cur_x, cur_y, cur_z);
				}
			}
		}
	}

	return;
}
