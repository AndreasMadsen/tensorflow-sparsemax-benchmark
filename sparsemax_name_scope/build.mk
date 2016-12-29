
# user friendly targets
sparsemax-name-scope-all: sparsemax-name-scope-test

sparsemax-name-scope-test:
	PYTHONPATH=./ python3 sparsemax_name_scope/python/kernel_tests/sparsemax_test.py
	PYTHONPATH=./ python3 sparsemax_name_scope/python/kernel_tests/sparsemax_loss_test.py
