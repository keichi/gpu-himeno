/********************************************************************

 This benchmark test program is measuring a cpu performance
 of floating point operation by a Poisson equation solver.

 If you have any question, please ask me via email.
 written by Ryutaro HIMENO, November 26, 2001.
 Version 3.0
 ----------------------------------------------
 Ryutaro Himeno, Dr. of Eng.
 Head of Computer Information Division,
 RIKEN (The Institute of Pysical and Chemical Research)
 Email : himeno@postman.riken.go.jp
 ---------------------------------------------------------------
 You can adjust the size of this benchmark code to fit your target
 computer. In that case, please chose following sets of
 (mimax,mjmax,mkmax):
 small : 33,33,65
 small : 65,65,129
 midium: 129,129,257
 large : 257,257,513
 ext.large: 513,513,1025
 This program is to measure a computer performance in MFLOPS
 by using a kernel which appears in a linear solver of pressure
 Poisson eq. which appears in an incompressible Navier-Stokes solver.
 A point-Jacobi method is employed in this solver as this method can
 be easyly vectrized and be parallelized.
 ------------------
 Finite-difference method, curvilinear coodinate system
 Vectorizable and parallelizable on each grid point
 No. of grid points : imax x jmax x kmax including boundaries
 ------------------
 A,B,C:coefficient matrix, wrk1: source term of Poisson equation
 wrk2 : working area, OMEGA : relaxation parameter
 BND:control variable for boundaries and objects ( = 0 or 1)
 P: pressure
********************************************************************/

#include <stdio.h>
#include <sys/time.h>

#include <algorithm>
#include <execution>
#include <experimental/mdspan>
#include <ranges>
#include <vector>

#ifdef SSMALL
#define MIMAX 33
#define MJMAX 33
#define MKMAX 65
#endif

#ifdef SMALL
#define MIMAX 65
#define MJMAX 65
#define MKMAX 129
#endif

#ifdef MIDDLE
#define MIMAX 129
#define MJMAX 129
#define MKMAX 257
#endif

#ifdef LARGE
#define MIMAX 257
#define MJMAX 257
#define MKMAX 513
#endif

#ifdef ELARGE
#define MIMAX 513
#define MJMAX 513
#define MKMAX 1025
#endif

namespace stdex = std::experimental;

void initmt(stdex::mdspan<float, stdex::dextents<size_t, 3>> p,
            stdex::mdspan<float, stdex::dextents<size_t, 4>> a,
            stdex::mdspan<float, stdex::dextents<size_t, 4>> b,
            stdex::mdspan<float, stdex::dextents<size_t, 4>> c,
            stdex::mdspan<float, stdex::dextents<size_t, 3>> bnd,
            stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk1,
            stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk2)
{
    auto r1 = std::views::iota(0, MIMAX * MJMAX * MKMAX);
    std::for_each(std::execution::par_unseq, r1.begin(), r1.end(), [=](int ijk) {
        int i = ijk / (MJMAX * MKMAX);
        int jk = ijk % (MJMAX * MKMAX);
        int j = jk / MKMAX;
        int k = jk % MKMAX;

        a(0, i, j, k) = 0.0f;
        a(1, i, j, k) = 0.0f;
        a(2, i, j, k) = 0.0f;
        a(3, i, j, k) = 0.0f;
        b(0, i, j, k) = 0.0f;
        b(1, i, j, k) = 0.0f;
        b(2, i, j, k) = 0.0f;
        c(0, i, j, k) = 0.0f;
        c(1, i, j, k) = 0.0f;
        c(2, i, j, k) = 0.0f;
        p(i, j, k) = 0.0f;
        wrk1(i, j, k) = 0.0f;
        bnd(i, j, k) = 0.0f;
    });

    auto r2 = std::views::iota(0, (MIMAX - 1) * (MJMAX - 1) * (MKMAX - 1));
    std::for_each(std::execution::par_unseq, r2.begin(), r2.end(), [=](int ijk) {
        int i = ijk / ((MJMAX - 1) * (MKMAX - 1));
        int jk = ijk % ((MJMAX - 1) * (MKMAX - 1));
        int j = jk / (MKMAX - 1);
        int k = jk % (MKMAX - 1);

        a(0, i, j, k) = 1.0f;
        a(1, i, j, k) = 1.0f;
        a(2, i, j, k) = 1.0f;
        a(3, i, j, k) = 1.0f / 6.0f;
        b(0, i, j, k) = 0.0f;
        b(1, i, j, k) = 0.0f;
        b(2, i, j, k) = 0.0f;
        c(0, i, j, k) = 1.0f;
        c(1, i, j, k) = 1.0f;
        c(2, i, j, k) = 1.0f;
        p(i, j, k) = (float)(i * i) / (float)((MIMAX - 2) * (MIMAX - 2));
        wrk1(i, j, k) = 0.0f;
        bnd(i, j, k) = 1.0f;
    });
}

float jacobi(int nn,
             stdex::mdspan<float, stdex::dextents<size_t, 3>> p,
             stdex::mdspan<float, stdex::dextents<size_t, 4>> a,
             stdex::mdspan<float, stdex::dextents<size_t, 4>> b,
             stdex::mdspan<float, stdex::dextents<size_t, 4>> c,
             stdex::mdspan<float, stdex::dextents<size_t, 3>> bnd,
             stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk1,
             stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk2)
{
    float gosa;
    float omega = 0.8f;

    for (int n = 0; n < nn; ++n) {
        auto r = std::views::iota(0, (MIMAX - 3) * (MJMAX - 3) * (MKMAX - 3));

        gosa = std::transform_reduce(std::execution::par_unseq, r.begin(), r.end() , 0.0, std::plus{}, [=](int ijk) {
            int i = ijk / ((MJMAX - 3) * (MKMAX - 3)) + 1;
            int jk = ijk % ((MJMAX - 3) * (MKMAX - 3));
            int j = jk / (MKMAX - 3) + 1;
            int k = jk % (MKMAX - 3) + 1;

            float s0 =
                a(0, i, j, k) * p(i + 1, j, k) +
                a(1, i, j, k) * p(i, j + 1, k) +
                a(2, i, j, k) * p(i, j, k + 1) +
                b(0, i, j, k) * (p(i + 1, j + 1, k) - p(i + 1, j - 1, k) -
                                 p(i - 1, j + 1, k) + p(i - 1, j - 1, k)) +
                b(1, i, j, k) * (p(i, j + 1, k + 1) - p(i, j - 1, k + 1) -
                                 p(i, j + 1, k - 1) + p(i, j - 1, k - 1)) +
                b(2, i, j, k) * (p(i + 1, j, k + 1) - p(i - 1, j, k + 1) -
                                 p(i + 1, j, k - 1) + p(i - 1, j, k - 1)) +
                c(0, i, j, k) * p(i - 1, j, k) +
                c(1, i, j, k) * p(i, j - 1, k) +
                c(2, i, j, k) * p(i, j, k - 1) + wrk1(i, j, k);

            float ss = (s0 * a(3, i, j, k) - p(i, j, k)) * bnd(i, j, k);

            wrk2(i, j, k) = p(i, j, k) + omega * ss;

            return ss * ss;
        });

        std::for_each(std::execution::par_unseq, r.begin(), r.end(), [=](int ijk) {
            int i = ijk / ((MJMAX - 3) * (MKMAX - 3)) + 1;
            int jk = ijk % ((MJMAX - 3) * (MKMAX - 3));
            int j = jk / (MKMAX - 3) + 1;
            int k = jk % (MKMAX - 3) + 1;

            p(i, j, k) = wrk2(i, j, k);
        });
    } /* end n loop */

    return gosa;
}

double fflop(int mx, int my, int mz)
{
    return ((double)(mz - 2) * (double)(my - 2) * (double)(mx - 2) * 34.0);
}

double mflops(int nn, double cpu, double flop)
{
    return (flop / cpu * 1.e-6 * (double)nn);
}

double second()
{
    struct timeval tm;
    double t;

    static int base_sec = 0, base_usec = 0;

    gettimeofday(&tm, NULL);

    if (base_sec == 0 && base_usec == 0) {
        base_sec = tm.tv_sec;
        base_usec = tm.tv_usec;
        t = 0.0;
    } else {
        t = (double)(tm.tv_sec - base_sec) +
            ((double)(tm.tv_usec - base_usec)) / 1.0e6;
    }

    return t;
}

int main(int argc, char *argv[])
{
    int nn;
    float gosa;
    double cpu, cpu0, cpu1, flop, target;

    target = 60.0;

    std::vector<float> p_v(MIMAX * MJMAX * MKMAX);
    std::vector<float> a_v(4 * MIMAX * MJMAX * MKMAX);
    std::vector<float> b_v(3 * MIMAX * MJMAX * MKMAX);
    std::vector<float> c_v(3 * MIMAX * MJMAX * MKMAX);
    std::vector<float> bnd_v(MIMAX * MJMAX * MKMAX);
    std::vector<float> wrk1_v(MIMAX * MJMAX * MKMAX);
    std::vector<float> wrk2_v(MIMAX * MJMAX * MKMAX);

    stdex::mdspan<float, stdex::dextents<size_t, 3>> p(p_v.data(), MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 4>> a(a_v.data(), 4, MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 4>> b(b_v.data(), 3, MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 4>> c(c_v.data(), 3, MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 3>> bnd(bnd_v.data(), MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk1(wrk1_v.data(), MIMAX, MJMAX, MKMAX);
    stdex::mdspan<float, stdex::dextents<size_t, 3>> wrk2(wrk2_v.data(), MIMAX, MJMAX, MKMAX);

    /*
     *    Initializing matrixes
     */
    initmt(p, a, b, c, bnd, wrk1, wrk2);
    printf("mimax = %d mjmax = %d mkmax = %d\n", MIMAX, MJMAX, MKMAX);
    printf("imax = %d jmax = %d kmax =%d\n", MIMAX - 1, MJMAX - 1, MKMAX - 1);

    nn = 3;
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n", nn);

    cpu0 = second();
    gosa = jacobi(nn, p, a, b, c, bnd, wrk1, wrk2);
    cpu1 = second();
    cpu = cpu1 - cpu0;

    flop = fflop(MIMAX - 1, MJMAX - 1, MKMAX - 1);

    printf(" MFLOPS: %f time(s): %f %e\n\n", mflops(nn, cpu, flop), cpu, gosa);

    nn = (int)(target / (cpu / 3.0));

    printf(" Now, start the actual measurement process.\n");
    printf(" The loop will be excuted in %d times\n", nn);
    printf(" This will take about one minute.\n");
    printf(" Wait for a while\n\n");

    /*
     *    Start measuring
     */
    cpu0 = second();
    gosa = jacobi(nn, p, a, b, c, bnd, wrk1, wrk2);
    cpu1 = second();

    cpu = cpu1 - cpu0;

    printf(" Loop executed for %d times\n", nn);
    printf(" Gosa : %e \n", gosa);
    printf(" MFLOPS measured : %f\tcpu : %f\n", mflops(nn, cpu, flop), cpu);
    printf(" Score based on Pentium III 600MHz : %f\n",
           mflops(nn, cpu, flop) / 82);

    return (0);
}

