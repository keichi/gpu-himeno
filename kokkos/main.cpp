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

#include <Kokkos_Core.hpp>

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

void initmt(Kokkos::View<float***, Kokkos::LayoutRight> p,
            Kokkos::View<float****, Kokkos::LayoutRight> a,
            Kokkos::View<float****, Kokkos::LayoutRight> b,
            Kokkos::View<float****, Kokkos::LayoutRight> c,
            Kokkos::View<float***, Kokkos::LayoutRight> bnd,
            Kokkos::View<float***, Kokkos::LayoutRight> wrk1,
            Kokkos::View<const float***, Kokkos::LayoutRight> wrk2)
{
    Kokkos::parallel_for(
        "initmt1",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                               {MIMAX, MJMAX, MKMAX}),
        KOKKOS_LAMBDA(int i, int j, int k) {
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

    Kokkos::parallel_for(
        "initmt2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {MIMAX - 1, MJMAX - 1, MKMAX - 1}),
        KOKKOS_LAMBDA(int i, int j, int k) {
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
             Kokkos::View<float***, Kokkos::LayoutRight> p,
             Kokkos::View<const float****, Kokkos::LayoutRight> a,
             Kokkos::View<const float****, Kokkos::LayoutRight> b,
             Kokkos::View<const float****, Kokkos::LayoutRight> c,
             Kokkos::View<const float***, Kokkos::LayoutRight> bnd,
             Kokkos::View<const float***, Kokkos::LayoutRight> wrk1,
             Kokkos::View<float***, Kokkos::LayoutRight> wrk2)
{
    float gosa;
    float omega = 0.8f;

    for (int n = 0; n < nn; ++n) {
        gosa = 0.0f;

        Kokkos::parallel_reduce(
            "jacobi1",
            Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Right>>(
                {1, 1, 1}, {MIMAX - 2, MJMAX - 2, MKMAX - 2}, {1, 16, 64}),
            KOKKOS_LAMBDA(int i, int j, int k, float &gosa) {
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

                gosa += ss * ss;

                wrk2(i, j, k) = p(i, j, k) + omega * ss;
            },
            gosa);

        Kokkos::parallel_for(
            "jacobi2",
            Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Right>>(
                {1, 1, 1}, {MIMAX - 2, MJMAX - 2, MKMAX - 2}, {1, 16, 64}),
            KOKKOS_LAMBDA(int i, int j, int k) { p(i, j, k) = wrk2(i, j, k); });

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

    Kokkos::ScopeGuard guard(argc, argv);

    Kokkos::View<float***, Kokkos::LayoutRight> p("p", MIMAX, MJMAX, MKMAX);
    Kokkos::View<float****, Kokkos::LayoutRight> a("a", 4, MIMAX, MJMAX, MKMAX);
    Kokkos::View<float****, Kokkos::LayoutRight> b("b", 3, MIMAX, MJMAX, MKMAX);
    Kokkos::View<float****, Kokkos::LayoutRight> c("c", 3, MIMAX, MJMAX, MKMAX);
    Kokkos::View<float***, Kokkos::LayoutRight> bnd("bnd", MIMAX, MJMAX, MKMAX);
    Kokkos::View<float***, Kokkos::LayoutRight> wrk1("wrk1", MIMAX, MJMAX, MKMAX);
    Kokkos::View<float***, Kokkos::LayoutRight> wrk2("wrk2", MIMAX, MJMAX, MKMAX);

    /*
     *    Initializing matrixes
     */
    initmt(p, a, b, c, bnd, wrk1, wrk2);
    Kokkos::fence();
    printf("mimax = %d mjmax = %d mkmax = %d\n", MIMAX, MJMAX, MKMAX);
    printf("imax = %d jmax = %d kmax =%d\n", MIMAX - 1, MJMAX - 1, MKMAX - 1);

    nn = 3;
    printf(" Start rehearsal measurement process.\n");
    printf(" Measure the performance in %d times.\n\n", nn);

    cpu0 = second();
    gosa = jacobi(nn, p, a, b, c, bnd, wrk1, wrk2);
    Kokkos::fence();
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
    Kokkos::fence();
    cpu1 = second();

    cpu = cpu1 - cpu0;

    printf(" Loop executed for %d times\n", nn);
    printf(" Gosa : %e \n", gosa);
    printf(" MFLOPS measured : %f\tcpu : %f\n", mflops(nn, cpu, flop), cpu);
    printf(" Score based on Pentium III 600MHz : %f\n",
           mflops(nn, cpu, flop) / 82);

    return (0);
}
