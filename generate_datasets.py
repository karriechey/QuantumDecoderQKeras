#!/usr/bin/env python3
"""Generate surface-code datasets via Stim, saving the arrays the training
pipelines consume (measurements, det_evts, flips). Mirrors the format of
Stim_generate_datasets.ipynb but is CLI-parameterized so rounds/distances can
be swept and the job can run in the background.

Files: data_d{d}_p{p:.3f}_r{rounds}.npz  (under --out-dir).
"""
import argparse, os, time
import numpy as np
import stim
from circuit_generators import get_builtin_circuit
from datetime import datetime


def gen_one(d, p, rounds, n_samples, seed, out_dir, measure_flip_prob):
    # 4-channel noise model matching Ulascan's surface_code_d5_r3_RCNN.ipynb:
    # before_measure_flip_probability is set to p there. (The earlier r=2 QAT
    # stand-in dataset used only the first 3 channels -- a different experiment.)
    circuit = get_builtin_circuit(
        'surface_code:rotated_memory_z',
        distance=d, rounds=rounds,
        before_round_data_depolarization=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p if measure_flip_prob is None else measure_flip_prob,
    )
    m_sampler = circuit.compile_sampler(seed=seed)
    measurements = m_sampler.sample(n_samples, bit_packed=False)
    converter = circuit.compile_m2d_converter()
    det_evts, flips = converter.convert(
        measurements=measurements, separate_observables=True, bit_packed=False)
    fn = f'{out_dir}/data_d{d}_p{p:.3f}_r{rounds}.npz'
    np.savez(fn, measurements=measurements, det_evts=det_evts, flips=flips)
    return fn, os.path.getsize(fn) / 1024**2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distances', type=int, nargs='+', default=[5])
    ap.add_argument('--probs', type=float, nargs='+',
                    default=[0.001, 0.005, 0.010, 0.050])
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--n-samples', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='./datasets')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--measure-flip-prob', type=float, default=None,
                    help='before_measure_flip_probability; default None = use p '
                         '(4-channel, matches his recipe). Set 0 for the old 3-channel family.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = [(d, p) for d in args.distances for p in args.probs]
    print(f'[gen] start {datetime.now():%Y-%m-%d %H:%M:%S}  '
          f'{len(configs)} configs  r={args.rounds}  N={args.n_samples:,}', flush=True)
    for i, (d, p) in enumerate(configs, 1):
        fn = f'{args.out_dir}/data_d{d}_p{p:.3f}_r{args.rounds}.npz'
        if os.path.exists(fn) and not args.overwrite:
            print(f'[gen] [{i}/{len(configs)}] skip (exists) {fn}', flush=True)
            continue
        t0 = time.time()
        fn, mb = gen_one(d, p, args.rounds, args.n_samples, args.seed,
                         args.out_dir, args.measure_flip_prob)
        print(f'[gen] [{i}/{len(configs)}] d={d} p={p:.3f} -> {fn} '
              f'({mb:.1f} MB, {time.time()-t0:.1f}s)', flush=True)
    print(f'[gen] done {datetime.now():%Y-%m-%d %H:%M:%S}', flush=True)


if __name__ == '__main__':
    main()
