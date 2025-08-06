# CP - arallel Computing for Molecular Dynamics
 
This repository contains solutions for the three stages of the **Parallel Computing** course project (MEI: Computação Paralela) at the University of Minho. The project focuses on optimizing a molecular dynamics simulation of argon gas atoms for performance using parallel computing techniques.

## Overview

The project is divided into three phases, gradually introducing performance optimization, shared-memory parallelism with OpenMP and GPU acceleration with CUDA. The main objective is to reduce execution time while maintaining correct and consistent simulation results.

## Phase 1 – Monothread Optimization

- Profiled the provided sequential code to identify bottlenecks.
- Applied low-level optimizations to reduce computation time (e.g., loop unrolling, memory access improvements).
- Ensured code readability and maintained output integrity.

## Phase 2 – Shared-Memory Parallelism with OpenMP

- Analyzed hotspots in the optimized sequential version.
- Applied OpenMP directives to parallelize compute-intensive sections.
- Evaluated speed-up and scalability by running the simulation across multiple cores.

## Phase 3 – GPU Acceleration with CUDA

- Chose CUDA for advanced parallelization to leverage GPU computing power.
- Adapted core simulation functions for execution on NVIDIA GPUs.
- Performed performance benchmarking and scalability analysis between CPU and GPU versions.

## Tools & Technologies

- **Languages**: C, CUDA 
- **Libraries**: OpenMP, CUDA Toolkit
- **Tools**: `gprof`, `nvprof`, `perf`, `valgrind`, `make`
