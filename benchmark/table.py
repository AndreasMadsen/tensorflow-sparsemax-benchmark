
import itertools

from tabulate import tabulate
import numpy as np
import scipy.stats


class SummaryTable:
    def __init__(self, content, col_names, row_names, **kwargs):
        formatted = self.content(content, row_names, **kwargs)

        self.format = tabulate(formatted, [''] + col_names, tablefmt="pipe")

    @staticmethod
    def content(content, row_names, format="%.3f ± %.4f"):
        # calculate statistics
        mean = np.mean(content, axis=2)
        sem = scipy.stats.sem(content, axis=2)
        ci = scipy.stats.t.interval(
            0.95, content.shape[2] - 1, scale=sem
        )[1]

        return [
            [name_row] + [
                format % (mean_val, ci_val)
                for mean_val, ci_val in zip(mean_row, ci_row)
            ] for name_row, mean_row, ci_row in zip(row_names, mean, ci)
        ]

    def __str__(self):
        return self.format
