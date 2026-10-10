CXX ?= g++
NVCC ?= nvcc
# Colab T4 is sm_75; override with: make ARCH=sm_80 etc. (use -arch=native if your nvcc supports it)
ARCH ?= sm_75

# default: everything (needs nvcc). `make cpu` builds only the CPU programs and works without a GPU.
all: bin/cpu_tc bin/cpu_densest bin/cpu_dsg bin/gpu_tc bin/gpu_densest bin/gpu_dsg

bin/cpu_tc: src/cpu_tc.cpp src/graph.h
	@mkdir -p bin
	$(CXX) -O3 -std=c++17 -fopenmp src/cpu_tc.cpp -o $@

# CPU-only densest-subgraph checker: cpu_batch_peel (serial + OpenMP) and cpu_greedy_peel
bin/cpu_densest: src/cpu_densest.cpp src/densest.h src/graph.h
	@mkdir -p bin
	$(CXX) -O3 -std=c++17 -fopenmp src/cpu_densest.cpp -o $@

# CPU reference of the paper algorithms (Greedy++, (k,Psi)-core decomposition, Exact, CoreExact) for triangle density
bin/cpu_dsg: src/cpu_dsg.cpp src/dsg.h src/densest.h src/graph.h
	@mkdir -p bin
	$(CXX) -O3 -std=c++17 -fopenmp src/cpu_dsg.cpp -o $@

cpu: bin/cpu_tc bin/cpu_densest bin/cpu_dsg

bin/gpu_tc: src/gpu_tc.cu src/graph.h src/kernels.cuh
	@mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp src/gpu_tc.cu -o $@

bin/gpu_densest: src/gpu_densest.cu src/densest.h src/graph.h src/kernels.cuh
	@mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp src/gpu_densest.cu -o $@

# GPU core decomposition + CPU CoreExact/Greedy++ pipeline (stage 6)
bin/gpu_dsg: src/gpu_dsg.cu src/dsg.h src/densest.h src/graph.h src/kernels.cuh
	@mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -Xcompiler -fopenmp src/gpu_dsg.cu -o $@

gpu: bin/gpu_tc bin/gpu_densest bin/gpu_dsg

clean:
	rm -rf bin

# CPU-only checks (no GPU): emulate every kernel on small graphs vs brute force
test:
	bash scripts/test_emu.sh
.PHONY: all cpu gpu clean test
