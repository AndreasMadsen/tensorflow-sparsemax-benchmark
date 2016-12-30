
import os.path as path

import numpy as np

import providers
import datasets
from table import SummaryTable

thisdir = path.dirname(path.realpath(__file__))
resultsdir = path.join(thisdir, '..', 'results')


def core_timings(providers, datasets,
                 repetitions=10, iterations=100, preheat_iterations=100,
                 verbose=False):
    col_names = [''] * len(providers)
    row_names = [''] * len(datasets) * 4
    results = np.zeros((len(datasets) * 4, len(providers), repetitions))

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
                    row_names[row_index] = "%s %s" % (
                        dataset.name, benchmarker_name
                    )

                    # preheat the graph
                    benchmarker(iterations=preheat_iterations)

                    # benchmark the graph
                    for repetition_i in range(repetitions):
                        took = benchmarker(iterations=iterations)
                        results[row_index, provider_i, repetition_i] = took

                        if verbose:
                            print('      %d: %f' % (repetition_i, took))

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
