#pragma once

enum class Modes {
	ALL_COMPONENTS,
	ALL_COMPLEX_COMPONENTS,
	ALL_REAL_COMPONENTS,
	ONE_COMPLEX_COMPONENT
};

const double EPSILON = 1e-10;

class Field {
private:
	double* vec_r[3];
	fftw_complex* vec_c[3];
	fftw_plan forward_plan[3], backward_plan[3];

	const Modes mode;

	const ptrdiff_t N;
	const double TAU, ETA;
	const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
	const double RANGE_LEFT, RANGE_RIGHT;

	const ptrdiff_t alloc_local, local_dim0_size, local_dim0_start;

	ptrdiff_t *indeces;
	const double NORMALIZATION_CONSTANT;
	const int rank, size;

	void cross_product(const Field&, const Field&);

	void divergence(const Field&);

public:
	Field(const Modes, const ptrdiff_t, const double, const double,
	      const double, const double,
	      const ptrdiff_t, const ptrdiff_t, const ptrdiff_t,
	      const int, const int);

	virtual ~Field();

	void forward_transformation();

	void backward_transformation();

	void correction(Field&);

	void rotor(const Field&, const Field&);

	void do_step(const Field&, Field&);

	double energy_phi() const;

	double energy_fourie() const;

	void fill_velocity_field();

	void fill_magnetic_field();
};
