
# sparsemax targets
sparsemax_kernel/python/ops/_sparsemax.so: sparsemax_kernel/ops/sparsemax.o sparsemax_kernel/kernels/sparsemax.o sparsemax_kernel/kernels/sparsemax_functor.o sparsemax_kernel/kernels/sparsemax_functor.cu.o

sparsemax_kernel/kernels/sparsemax_functor.o: sparsemax_kernel/kernels/sparsemax_functor.cc sparsemax_kernel/kernels/sparsemax_functor.h
sparsemax_kernel/kernels/sparsemax_functor.cu.o: sparsemax_kernel/kernels/sparsemax_functor.cu.cc sparsemax_kernel/kernels/sparsemax_functor.h

sparsemax_kernel/python/ops/_sparsemax_loss.so: sparsemax_kernel/ops/sparsemax_loss.o sparsemax_kernel/kernels/sparsemax_loss.o sparsemax_kernel/kernels/sparsemax_loss.cu.o

sparsemax_kernel/kernels/sparsemax_loss.o: sparsemax_kernel/kernels/sparsemax_loss.cc sparsemax_kernel/kernels/sparsemax_loss.h
sparsemax_kernel/kernels/sparsemax_loss.cu.o: sparsemax_kernel/kernels/sparsemax_loss.cu.cc sparsemax_kernel/kernels/sparsemax_loss.h


# user friendly targets
sparsemax-kernel-all: sparsemax-kernel-build sparsemax-kernel-test

sparsemax-kernel-build: sparsemax_kernel/python/ops/_sparsemax.so sparsemax_kernel/python/ops/_sparsemax_loss.so

sparsemax-kernel-test: sparsemax-kernel-build
	PYTHONPATH=./ python3 sparsemax_kernel/python/kernel_tests/sparsemax_test.py
	PYTHONPATH=./ python3 sparsemax_kernel/python/kernel_tests/sparsemax_loss_test.py
