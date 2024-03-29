"""
ising model simulator
"""
import argparse
import datetime
import functools
#import glob
import itertools
import os
import pickle
#import re
#import sys
import time
from multiprocessing import Pool

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#plt.style.use(['science', 'notebook', 'no-latex'])
import numpy as np
import clr
clr.AddReference('Simulator_R2')
from Simulator import Ising2d, Ising2dExchange


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", default='single', choices=['single', 'exchange', 'exchange-parallel'],
    help="2d ising model size")
parser.add_argument(
    "-s", "--size", nargs=2, default=(32, 32), type=int,
    help="2d ising model size")
parser.add_argument(
    "-n", "--num-samples", default=1000, type=int,
    help="number of simulation samples")
parser.add_argument(
    "-b", "--beta", nargs=3, default=(0.35, 0.55, 0.01), type=float,
    help="betas to do calclation (start, stop ,step)")
parser.add_argument(
    "-r", "--rec-num", default=3, type=int,
    help="number of recurring each beta")
parser.add_argument(
    "--bootstrap", default=100, type=int,
    help="bootstrap number to calclate specific heat")
parser.add_argument(
    "-c", "--cpu-use", type=int,
    help="number of use cpu (default: number of betas)")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-v", "--verbose", action="count", default=0,
    help="increase output verbosity")
group.add_argument(
    "-q", "--quiet", action="store_true", default=0,
    help="decrease output verbosity")

abshere = os.path.dirname(os.path.abspath(__file__))


def log_format(obj, float_digit=3):
    if isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, float):
        return format(round(obj, float_digit))
    elif hasattr(obj, "__iter__"):
        return "[" + ", ".join(log_format(item) for item in obj) + "]"
    else:
        return str(obj)

def pos_arg_count(func):
    return len(func.__code__.co_argcount) - len(func.__defaults__)

def arg_dic(func, args):
    return {k:v for k,v in zip(func.__code__.co_varnames, args)}

def kwargs_complete(func, kwargs):
    for kw,val in zip(func.__code__.co_argment[pos_arg_count(func):], func.__defaults__):
        if kw not in kwargs.keys():
            kwargs[kw] = val
    return kwargs

def timer(func):
    """
    参考: https://qiita.com/hisatoshi/items/7354c76a4412dffc4fd7
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        results = func(*args, **kwargs)
        elapsed_time =  datetime.datetime.now() - start
        return results, elapsed_time
    return wrapper

def output_string(element2d, sep1=" ", sep2=", "):
    #width = os.get_terminal_size().columns - max(len(sep1), len(sep2))
    width=200
    init_tab = len(element2d[0][0]) + 1
    output_str = ""
    output_col = 0
    for elements in element2d:
        for element in elements:
            le = len(element)
            if output_col + len(sep2) + le > width:
                output_str += "\n" + " " * init_tab + element + sep2
                output_col = le + init_tab
            else:
                output_str += element + sep2
                output_col += le + len(sep2)
        output_str = output_str[:-len(sep2)] + sep1
        output_col += len(sep1) - len(sep2)
    return output_str[:-len(sep1)]

def io_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        output_str = output_string([["start:"],
            [f'{k}: {log_format(v)}'
                for k,v in (arg_dic(func, args) | kwargs).items()],
            ["at " + start.strftime("%H:%M:%S.%f")[:-3]]])
        print(output_str)

        results = func(*args, **kwargs)
        elapsed_time =  datetime.datetime.now() - start

        output_str = output_string([["end:"],
            [log_format(x)
                for x in (results if isinstance(results, tuple) else [results])],
            ["in " + str(elapsed_time)]])
        print(output_str)
        return results
    return wrapper

def unpack_exe(func):
    @functools.wraps(func)
    def wrapper(args):
        if isinstance(args, dict):
            return func(**args)
        elif hasattr(args, "__iter__"):
            return func(*args)
    return wrapper


@unpack_exe
@io_logger
def ising_sim(beta, width, height, sample_size, bootstrap, verbose):
    """hoge"""
    # = params
    save_dir = os.path.join("save_data", f"ising_{width}x{height}_n-{sample_size}")

    time.sleep(np.random.rand())
    #start = datetime.datetime.now()
    sim = Ising2d(width, height, beta)
    sim.Sampling(sample_size)
    #if verbose:
    #    print("start:", f"{width}, {height}, {sample_size}, {beta:.3f}")
    res = sim.SpecificHeats(sample_size - sample_size / 10, 10, bootstrap)
    spe_mean, spe_std = res.Item1, res.Item2
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
    with open(os.path.join(save_dir,
            f"b-{beta:.3f}_t-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.bin"), "wb") as file:
        pickle.dump((
            spe_mean, spe_std,
            list(sim.Interactions),
            list(sim.Magnetizations),
            list(sim.Energys)
        ), file)
    #if verbose:
    #    print("end:", f"{beta:.3f}, {spe_mean:.3f}, {spe_std:.3f}, "\
    #        f"{(datetime.datetime.now() - start)}")
    return spe_mean, spe_std


@unpack_exe
@io_logger
def ising_sim_ex(betas, width, height, sample_size, bootstrap, verbose):
    """hoge"""
    # = params
    save_dir = os.path.join("save_data",
        f"ising_ex_{width}x{height}_n-{sample_size}_bn-{len(betas)}")
    burnin = sample_size - sample_size//10
    skip = 10

    time.sleep(np.random.rand())
    #start = datetime.datetime.now()
    #if verbose>=1:
    #    print("start:", f"{width}, {height}, {sample_size}, {len(betas)}")
    sim = Ising2dExchange(width, height, betas, 10)
    sim.Sampling(sample_size)
    energy_n_k = np.array(list(sim.GetEnergys()))

    ll_means = -np.array(betas) * energy_n_k[:, burnin::skip].mean(axis=-1)
    res = sim.SpecificHeat_k(burnin, skip, bootstrap)
    spe_means, spe_stds = np.array(list(res.Item1)), np.array(list(res.Item2))
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
    with open(os.path.join(save_dir,
            f"t-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.bin"), "wb") as file:
        interaction_n_k = np.array(list(sim.GetInteractions()))
        magnetization_n_k = np.array(list(sim.GetMagnetization()))
        chain_index_n_k = np.array(list(sim.BetaIndex_n_k))
        pickle.dump((
            spe_means, spe_stds, np.array(betas),
            interaction_n_k,
            magnetization_n_k,
            energy_n_k,
            chain_index_n_k
        ), file)
    #if verbose>=1:
    #    print("end:", f"{width}, {height}, {(datetime.datetime.now() - start)}")
    #if verbose>=2:
    #    print("\n".join([f"\t{beta:.3f}, {mean:.3f}, {std:.3f}, {ll_mean:.3f}"
    #        for beta,mean,std,ll_mean in zip(betas, spe_means, spe_stds, ll_means)]))
    return spe_means, spe_stds, ll_means


if __name__=="__main__":
    args = parser.parse_args()
    mode = args.mode
    verbosity = 1 + args.verbose - args.quiet
    if verbosity>=2:
        print(args)
    if verbosity>=3:
        exit()
    width_, height_ = args.size
    sample_size_ = args.num_samples
    rec_num_ = args.rec_num
    bootstrap_ = args.bootstrap
    a,b,c = args.beta
    betas_ = np.arange(a,b+c,c)
    cpu_use = args.cpu_use
    start_ = datetime.datetime.now()
    if mode=='single':
        if cpu_use is None:
            cpu_use = len(betas_)
        if cpu_use > os.cpu_count():
            cpu_use = os.cpu_count()
        if verbosity>=2:
            print("cpu count:", cpu_use)
        with Pool(cpu_use) as p:
            result = p.map_async(
                ising_sim,
                [(beta, width_, width_, sample_size_, bootstrap_, verbosity)
                    for beta in betas_ for _ in range(rec_num_)]).get(10**6)
        if verbosity>=2:
            betas_ = [beta for beta in betas_ for _ in range(3)]
            for beta_, (mean, std) in zip(betas_, result):
                print(f"{beta_:.3f},\t{mean:.3f},\t{std:.3f}")
            print("total time:", datetime.datetime.now() - start_)
    elif mode=='exchange':
        if cpu_use is None:
            cpu_use = rec_num_
        if cpu_use > os.cpu_count():
            cpu_use = os.cpu_count()
        if verbosity>=2:
            print("cpu count:", cpu_use)
        with Pool(cpu_use) as p:
            result = p.map_async(
                ising_sim_ex,
                [(betas_, width_, width_, sample_size_, bootstrap_, verbosity)
                    for _ in range(rec_num_)]).get(10**6)
        if verbosity>=2:
            mean_k_r, std_k_r, _ll_means_k_r = np.array(result).transpose((1,0,2))
            for beta_, means, stds in zip(betas_, mean_k_r.T, std_k_r.T):
                #print(beta, means, stds)
                mean = means.mean()
                std = np.sqrt( (((bootstrap_-1)*stds**2).sum() + (bootstrap_*(means-mean)**2).sum())
                     / (bootstrap_*3-1) )
                print(f"{beta_:.3f},\t{mean:.3f},\t{std:.3f}")
            print("total time:", datetime.datetime.now() - start_)
    elif mode=='exchange-parallel':
        if cpu_use is None or cpu_use > os.cpu_count():
            cpu_use = os.cpu_count()
        if verbosity>=2:
            print("cpu count:", cpu_use)
        raise NotImplementedError("'exchange-parallel' module has not yet been implemented.")
