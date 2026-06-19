#!/usr/bin/env python3
"""Systematic benchmark of Ulascan's FullRCNNModel (his real architecture from
CNNModel.py) across configs and random seeds, with an MWPM baseline.

This is Pillar 1: the foundation the paper needs before quantization. It records
per-seed logical error rates (so error bars are computable), training time, and
saves trained weights (so we don't retrain each time -- a reviewer ask).

Usage (local d=5 test):
  python benchmark_rcnn.py --distances 5 --probs 0.01 --rounds 3 \
      --seeds 0 1 2 --n-train 100000 --n-test 50000 --epochs 30

Reads datasets/data_d{d}_p{p:.3f}_r{rounds}.npz (measurements, det_evts, flips).
Writes results/benchmark_rcnn.csv and results/weights/<config>_seed<s>.weights.h5.
"""
import argparse, csv, os, platform, time
import numpy as np


def set_seeds(seed):
    import random, tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def learning_rate_scheduler(epoch, lr):
    """Ulascan's schedule from surface_code_d5_r3_RCNN.ipynb. Starts high
    (0.01 at epoch 0) to move the slow-start zero-init state correlator off the
    ground, then decays. A constant 0.001 Adam crawls and underfits."""
    if epoch < 10:
        return 0.001 * (10 - epoch)
    elif epoch < 20:
        return lr * 0.9
    elif epoch < 30:
        return lr * 0.8
    else:
        return lr * 0.65


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument('--distances', type=int, nargs='+', default=[5])
    ap.add_argument('--probs', type=float, nargs='+', default=[0.01])
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--n-train', type=int, default=100_000)
    ap.add_argument('--n-test', type=int, default=50_000)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--hidden', type=int, default=100)
    ap.add_argument('--hidden-layers', type=int, default=2)
    ap.add_argument('--npol', type=int, default=2)
    ap.add_argument('--data-dir', default='./datasets')
    ap.add_argument('--results-csv', default='./results/benchmark_rcnn.csv')
    ap.add_argument('--weights-dir', default='./results/weights')
    ap.add_argument('--no-save-weights', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    import pymatching
    from types_cfg import get_types
    from circuit_generators import get_builtin_circuit
    from circuit_partition import split_measurements
    from CNNModel import FullRCNNModel

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    if not args.no_save_weights:
        os.makedirs(args.weights_dir, exist_ok=True)

    host = f"{platform.system()}-{platform.machine()}"
    tf_ver = tf.__version__
    fields = ['architecture', 'd', 'p', 'rounds', 'kernel', 'seed', 'n_train',
              'n_test', 'epochs', 'epochs_ran', 'batch_size', 'n_params', 'p_L',
              'mwpm_p_L', 'best_val_loss', 'train_time_s', 'host', 'tf_version']
    new_file = not os.path.exists(args.results_csv)
    csv_f = open(args.results_csv, 'a', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=fields)
    if new_file:
        writer.writeheader()

    obs = 'ZL'  # rotated_memory_z
    configs = [(d, p) for d in args.distances for p in args.probs]
    for d, p in configs:
        binary_t, time_t, idx_t, packed_t = get_types(d, args.rounds, args.kernel)
        fn = f'{args.data_dir}/data_d{d}_p{p:.3f}_r{args.rounds}.npz'
        if not os.path.exists(fn):
            print(f'[bench] MISSING {fn} -- run generate_datasets.py first; skipping', flush=True)
            continue
        z = np.load(fn)
        measurements = z['measurements'].astype(binary_t)
        det_evts = z['det_evts'].astype(binary_t)
        flips = z['flips'].astype(binary_t)
        need = args.n_train + args.n_test
        if measurements.shape[0] < need:
            print(f'[bench] {fn} has {measurements.shape[0]} < {need} samples; using what exists', flush=True)
        det_bits, _, _ = split_measurements(measurements, d, idx_t)

        # Fixed train/test split (independent of model seed) so all seeds see
        # the same data and the comparison is apples-to-apples.
        ntr, nte = args.n_train, args.n_test
        tr = slice(0, ntr); te = slice(ntr, ntr + nte)

        # MWPM baseline (once per config) on the same test shots.
        # Must match the 4-channel noise model the data was generated with
        # (generate_datasets.py sets before_measure_flip_probability=p), or the
        # DEM -> MWPM baseline is decoding a different circuit than the data.
        circ = get_builtin_circuit('surface_code:rotated_memory_z', distance=d,
            rounds=args.rounds, before_round_data_depolarization=p,
            after_reset_flip_probability=p, after_clifford_depolarization=p,
            before_measure_flip_probability=p)
        dem = circ.detector_error_model(decompose_errors=True)
        pym = pymatching.Matching.from_detector_error_model(dem)
        pp = pym.decode_batch(det_evts[te], bit_packed_predictions=False,
                              bit_packed_shots=False).astype(binary_t).reshape(-1, 1)
        mwpm_pL = float((pp != flips[te]).mean())

        per_seed = []
        for seed in args.seeds:
            set_seeds(seed)
            model = FullRCNNModel(obs, d, args.kernel, args.rounds,
                [args.hidden for _ in range(args.hidden_layers)], npol=args.npol,
                stop_round=None, has_nonuniform_response=False,
                do_all_data_qubits=False, return_all_rounds=False)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            _ = model([det_bits[0:1], det_evts[0:1]])  # build
            n_params = int(model.count_params())
            t0 = time.time()
            hist = model.fit(x=[det_bits[tr], det_evts[tr]], y=flips[tr],
                      batch_size=args.batch_size, epochs=args.epochs,
                      validation_split=args.val_split, verbose=2,
                      callbacks=[
                          tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              patience=args.patience, restore_best_weights=True),
                          tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler),
                      ])
            train_time = time.time() - t0
            epochs_ran = len(hist.history['loss'])
            best_val = float(min(hist.history['val_loss']))
            pred = model.predict([det_bits[te], det_evts[te]],
                                 batch_size=args.batch_size, verbose=0)
            pL = float((flips[te] != (pred > 0.5).astype(binary_t)).mean())
            per_seed.append(pL)
            if not args.no_save_weights:
                tag = f'rcnn_d{d}_p{p:.3f}_r{args.rounds}_k{args.kernel}_seed{seed}'
                model.save_weights(f'{args.weights_dir}/{tag}.weights.h5')
            writer.writerow(dict(architecture='FullRCNNModel', d=d, p=p,
                rounds=args.rounds, kernel=args.kernel, seed=seed, n_train=ntr,
                n_test=nte, epochs=args.epochs, epochs_ran=epochs_ran,
                batch_size=args.batch_size, n_params=n_params, p_L=pL,
                mwpm_p_L=mwpm_pL, best_val_loss=round(best_val, 5),
                train_time_s=round(train_time, 1), host=host, tf_version=tf_ver))
            csv_f.flush()
            print(f'[bench] d={d} p={p:.3f} seed={seed}  RCNN p_L={pL:.4f}  '
                  f'MWPM={mwpm_pL:.4f}  ({train_time:.0f}s, {n_params} params)', flush=True)
        arr = np.array(per_seed)
        print(f'[bench] === d={d} p={p:.3f}: RCNN p_L = {arr.mean():.4f} '
              f'± {arr.std(ddof=1) if len(arr)>1 else 0:.4f}  (n={len(arr)} seeds)  '
              f'| MWPM {mwpm_pL:.4f} ===', flush=True)
    csv_f.close()
    print(f'[bench] results -> {args.results_csv}', flush=True)


if __name__ == '__main__':
    run()
