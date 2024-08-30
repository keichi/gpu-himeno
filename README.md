# gpu-himeno

The famous Himeno benchmark ported to various GPU programming models.

## CUDA

```
nvcc -DELARGE --cudart shared -Xcompiler -mcmodel=large himenoBMT.cu
```

- 690 GFLOPS on A100 80GB
- 630 GFLOPS on A100 40GB

## HIP

```
hipcc -DELARGE -mcmodel=medium himenoBMT.cpp
```

- 509 GFLOPS on MI210
- 451 GFLOPS on MI100

## Kokkos

```
mkdir build && cd build
cmake -DKokkos_DIR=<path/to/kokkos> ..
make
```

- 546 GFLOPS on A100 80GB
- 487 GFLOPS on A100 40GB

## OpenACC

```
nvc++ -DELARGE -acc=gpu -mcmodel=medium himenoBMT.cpp
```

- 651 GFLOPS on A100 80GB
- 571 GFLOPS on A100 40GB

## Parallel STL

```
nvc++ -DELARGE -std=c++23 --gcc-toolchain=<path/to/toolchain> -stdpar=gpu --experimental-stdpar himenoBMT.cpp
```

- 468 GFLOPS on A100 80GB
- 411 GFLOPS on A100 40GB
