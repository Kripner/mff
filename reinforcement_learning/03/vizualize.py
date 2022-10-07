#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import numpy as np

import matplotlib.pyplot as plt

from q_learning import ValueInInterval

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--plot", dest="plots", action="append", nargs="+", type=str, help="Timeseries to plot.")
parser.add_argument("--prefix", type=str, default="", help="Prepend this to every name.")
parser.add_argument("--show", default=True, action="store_true", help="Show the plot.")
parser.add_argument("--save", type=str, help="Where to save the plot.")


@dataclass
class Timeseries:
    x: np.ndarray[int]
    y: np.ndarray[float]
    name: str


@dataclass
class TimeseriesWithInterval:
    x: np.ndarray[int]
    y: np.ndarray[float]
    lower: np.ndarray[float]
    upper: np.ndarray[float]
    name: str


def load_timeseries(name: str) -> Timeseries:
    last_slash_idx = name.rfind("/")
    if last_slash_idx == -1:
        raise Exception(f"Name {name} doesn't contain a path.")
    assert last_slash_idx != -1
    path, param = name[:last_slash_idx] + ".log", name[last_slash_idx + 1:]
    x, y = [], []
    has_interval = None
    lower, upper = [], []
    with open(path) as f:
        header = f.readline().split()
        idx = header.index(param)
        if idx == -1:
            raise Exception(f"Parameter {param} not contained in the file {path}.")
        for line in f:
            if line.startswith("#"):
                continue

            params = line.split()
            val_str = params[idx]

            if val_str != "-":
                x.append(int(params[0]))

                vii = ValueInInterval.try_from_string(val_str)
                if has_interval is None:
                    has_interval = vii is not None
                else:
                    assert has_interval == (vii is not None)

                if has_interval:
                    y.append(vii.value)
                    lower.append(vii.lower)
                    upper.append(vii.upper)
                else:
                    y.append(float(val_str))

    return Timeseries(np.array(x), np.array(y), name) \
        if not has_interval \
        else TimeseriesWithInterval(np.array(x), np.array(y), np.array(lower), np.array(upper), name)


def create_plot(plot):
    ax = plt.subplot()
    for series in plot:
        if isinstance(series, Timeseries):
            ax.plot(series.x, series.y, label=series.name)
        elif isinstance(series, TimeseriesWithInterval):
            ax.plot(series.x, series.y, label=series.name)
            ax.fill_between(series.x, series.lower, series.upper, alpha=0.2)
        else:
            assert False
    ax.legend(loc="lower right")
    plt.show()


def main():
    args = parser.parse_args([] if "__file__" not in globals() else None)
    plots = args.plots
    if len(args.prefix) > 0:
        prefix = args.prefix + "/" if args.prefix[-1] != "/" else args.prefix
        plots = list(map(lambda plot:
                         list(map(lambda name: prefix + name, plot)),
                         plots))
    assert len(plots) == 1
    plots = map(lambda plot:
                list(map(lambda name: load_timeseries(name), plot)),
                plots)
    create_plot(list(plots)[0])


if __name__ == "__main__":
    main()
