
import os
import time

import tensorflow as tf

import sparsemax_defun
import sparsemax_kernel
import sparsemax_name_scope


class _TensorflowProvider:
    def __init__(self, dataset, sparsemax, sparsemax_loss,
                 name='NoName', dtype=tf.float32, cpu_only=False):
        self.name = name
        self.dataset = dataset
        self.benchmark_names = [
            'sparsemax', 'sparsemax_grad',
            'sparsemax_loss', 'sparsemax_loss_grad'
        ]
        self.benchmarkers = [
            self.sparsemax, self.sparsemax_grad,
            self.sparsemax_loss, self.sparsemax_loss_grad
        ]
        self.cpu_only = cpu_only

        self.graph = tf.Graph()

        with self.graph.as_default():
            # setup inputs
            self.logits = tf.placeholder(
                dtype, [dataset.obs, dataset.dims], name='x'
            )
            self.labels = tf.placeholder(
                dtype, [dataset.obs, dataset.dims], name='t'
            )

            logits = self.logits_var = tf.Variable(
                self.logits, trainable=False, collections=[]
            )
            labels = self.labels_var = tf.Variable(
                self.labels, trainable=False, collections=[]
            )

            # setup functions
            self._sparsemax = sparsemax(logits)
            self._sparsemax_grad = tf.gradients(self._sparsemax, [logits])[0]

            self._sparsemax_loss = sparsemax_loss(
                logits, self._sparsemax, labels
            )
            self._sparsemax_loss_grad = tf.gradients(
                self._sparsemax_loss, [logits]
            )[0]

    def __enter__(self):
        # create session and reset variables
        config = None
        if self.cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        self._sess = tf.Session(graph=self.graph, config=config)

        # intialize variables
        self._sess.run([
            self.logits_var.initializer,
            self.labels_var.initializer
        ], feed_dict={
            self.logits: self.dataset.logits,
            self.labels: self.dataset.labels
        })

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # close session
        self._sess.close()

    def __iter__(self):
        return iter(zip(self.benchmark_names, self.benchmarkers))

    def sparsemax(self, iterations=1):
        tick = time.perf_counter()
        for _ in range(iterations):
            self._sess.run(self._sparsemax)
        return time.perf_counter() - tick

    def sparsemax_grad(self, iterations=1):
        tick = time.perf_counter()
        for _ in range(iterations):
            self._sess.run(self._sparsemax_grad)
        return time.perf_counter() - tick

    def sparsemax_loss(self, iterations=1):
        tick = time.perf_counter()
        for _ in range(iterations):
            self._sess.run(self._sparsemax_loss)
        return time.perf_counter() - tick

    def sparsemax_loss_grad(self, iterations=1):
        tick = time.perf_counter()
        for _ in range(iterations):
            self._sess.run(self._sparsemax_loss_grad)
        return time.perf_counter() - tick


class SparsemaxKernelCPU(_TensorflowProvider):
    def __init__(self, dataset):
        super().__init__(
            dataset,
            sparsemax=sparsemax_kernel.sparsemax,
            sparsemax_loss=sparsemax_kernel.sparsemax_loss,
            name='kernel CPU',
            cpu_only=True)


class SparsemaxKernelGPU(_TensorflowProvider):
    def __init__(self, dataset):
        super().__init__(
            dataset,
            sparsemax=sparsemax_kernel.sparsemax,
            sparsemax_loss=sparsemax_kernel.sparsemax_loss,
            name='kernel GPU',
            cpu_only=False)


class SparsemaxNameScopeCPU(_TensorflowProvider):
    def __init__(self, dataset):
        super().__init__(
            dataset,
            sparsemax=sparsemax_name_scope.sparsemax,
            sparsemax_loss=sparsemax_name_scope.sparsemax_loss,
            name='name_scope CPU',
            cpu_only=True)


class SparsemaxDefunCPU(_TensorflowProvider):
    def __init__(self, dataset):
        super().__init__(
            dataset,
            sparsemax=sparsemax_defun.sparsemax,
            sparsemax_loss=sparsemax_defun.sparsemax_loss,
            name='Defun CPU',
            cpu_only=True)


all_providers = [
    SparsemaxKernelCPU,
    SparsemaxKernelGPU,
    SparsemaxNameScopeCPU,
    SparsemaxDefunCPU
]

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    all_providers.remove(SparsemaxKernelGPU)
