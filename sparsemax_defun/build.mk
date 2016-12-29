
# user friendly targets
sparsemax-defun-all: sparsemax-defun-test

sparsemax-defun-test:
	PYTHONPATH=./ python3 sparsemax_defun/python/kernel_tests/sparsemax_test.py
	PYTHONPATH=./ python3 sparsemax_defun/python/kernel_tests/sparsemax_loss_test.py
