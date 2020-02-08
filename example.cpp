#include <mpi.h>
#include <complex>
#include <cmath>

using complexd = std::complex<double>;

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
    const int power_of_two;
    const ptrdiff_t N;
    const ptrdiff_t INDEX_LEFT, INDEX_RIGHT;
    const double RANGE_LEFT, RANGE_RIGHT;
    Task_features(  const int pwr_of_two_,
                    const double rng_left_, const double rng_right_):
                    power_of_two{pwr_of_two_},
                    RANGE_LEFT{rng_left_}, RANGE_RIGHT{rng_right_}
    {
        N = 1 << power_of_two;
        INDEX_LEFT = -N / 2 + 1, INDEX_RIGHT = N / 2;
    }
};

void derived(fftw_complex* ptr, const Task_features& info, const int num_of_dimension num) {
    switch(num_of_dimension) {
        case 0:
            for (ptrdiff_t i = info.INDEX_LEFT, i <= info.INDEX_RIGHT; ++i) {
                for ()
            }


            break;
        case 1:


            break;
        case 2:


            break;
        default:
            std::cerr << "AVOST" << '\n';
            exit(1);
    }
    return;
}

int main(int argc, char *argv[]) {
    const Task_features info(4, 0, std::acos(-1) * 2);

    fftw_plan plan;
    double *rin;
    fftw_complex *cout;
    ptrdiff_t alloc_local, local_n0, local_0_start, i, j, k;

    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* get local data size and allocate */
    alloc_local = fftw_mpi_local_size_3d(L, M, N / 2 + 1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);
    rin = fftw_alloc_real(2 * alloc_local);
    cout = fftw_alloc_complex(alloc_local);

    /* create plan for out-of-place r2c DFT */
    plan = fftw_mpi_plan_dft_r2c_3d(L, M, N, rin, cout, MPI_COMM_WORLD,
                                    FFTW_MEASURE);

    /* initialize rin to some function my_func(x,y,z) */
    for (i = 0; i < local_n0; ++i)
       for (j = 0; j < M; ++j)
         for (k = 0; k < N; ++k)
         rin[(i*M + j) * (2*(N/2+1)) + k] = i * j + k;

    /* compute transforms as many times as desired */
    fftw_execute(plan);



    fftw_free(rin);
    fftw_free(cout);
    fftw_destroy_plan(plan);

    MPI_Finalize();

    return 0;
}
