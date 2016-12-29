
import time
import os.path as path

import numpy as np

import providers
import datasets
from table import SummaryTable

thisdir = path.dirname(path.realpath(__file__))
resultsdir = path.join(thisdir, '..', 'results')


def core_timings(providers, datasets,
                 iterations=10, epochs=100, verbose=False):
    col_names = [''] * len(providers)
    row_names = [''] * len(datasets) * 4
    results = np.zeros((len(datasets) * 4, len(providers), iterations))

    for dataset_i, dataset in enumerate(datasets):
        if verbose:
            print(dataset.name)

        for provider_i, Provider in enumerate(providers):
            with Provider(dataset) as provider:
                if verbose:
                    print('  ' + provider.name)

                col_names[provider_i] = provider.name

                for benchmarker_i, (benchmarker_name, benchmarker) \
                        in enumerate(provider):
                    if verbose:
                        print('    ' + benchmarker_name)

                    row_index = dataset_i * 4 + benchmarker_i
                    row_names[row_index] = "%s %s" % (dataset.name, benchmarker_name)

                    for iteration_i in range(iterations):
                        tick = time.perf_counter()
                        benchmarker(epochs=epochs)
                        tock = time.perf_counter() - tick
                        results[row_index, provider_i, iteration_i] = tock

                        if verbose:
                            print('      %d: %f' % (iteration_i, tock))

    return (results, col_names, row_names)


def main():
    data, col_names, row_names = core_timings(
        providers.all_providers, datasets.all_datasets, verbose=True
    )
    np.savez(
        path.join(resultsdir, 'timings.npz'),
        data=data, col_names=col_names, row_names=row_names
    )

    table = SummaryTable(data, col_names, row_names)
    print(table)

if __name__ == "__main__":
    main()
