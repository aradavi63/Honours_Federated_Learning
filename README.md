# Federated Learning Privacy Experiments

Comparison of FedAvg, SIHEFL, DP-FedAvg and DP-SIHEFL on MNIST and CIFAR-10, evaluating gradient inversion, membership inference, label poisoning, model utility and execution costs.

## Setup

Set up the Conda environment from the repository root. Datasets download automatically. Results go into `results/`.

```powershell
conda env create -f environments/environment.yml
conda activate fl-security
```

## Running experiments

### Training

```powershell
python -m experiments.train --scheme fedavg --dataset MNIST `
  --local-epochs 5 --alpha 1
```

Available schemes: `fedavg`, `sihefl`, `dp_fedavg` and `dp_sihefl`.

Use `--dataset CIFAR10` for CIFAR-10. Each command accepts `--help` for additional options.

### Gradient inversion

```powershell
python -m experiments.gia --dataset MNIST `
  --schemes fedavg sihefl dp_fedavg dp_sihefl `
  --rounds 1 --num-users 3 --samples-per-client 4 `
  --local-batch-size 2 --target-client 1 --alpha 1 `
  --iterations 2400 --noise-multiplier 2.5
```

### Membership inference

```powershell
python -m experiments.membership --dataset MNIST `
  --schemes fedavg sihefl dp_fedavg dp_sihefl `
  --local-epochs 5 --alpha 1 --noise-multiplier 2.5
```

### Label poisoning

```powershell
python -m experiments.poisoning --dataset MNIST --suite `
  --schemes fedavg sihefl dp_fedavg dp_sihefl `
  --num-poisoned-clients 2 --noise-multiplier 2.5
```

`--suite` runs the clean control and all label mappings for the selected dataset.

## Results

Results are saved under `results/`.

| File | Contents |
| --- | --- |
| `utility.csv` | Model accuracy and loss by round |
| `cost.csv` | Runtime, payload sizes and transmission counts |
| `procedure_times.csv` | Native procedure timings by round |
| `privacy.json` | Privacy accounting for DP schemes |
| Attack-specific CSV files | Attack scores and evaluation metrics |

Settings are saved in `configuration.json`, or `attack_config.json` for poisoning. Poisoning also keeps `status.json` for resuming completed runs.

Gradient inversion saves `per_image.csv`, `comparison.png` and `reconstruction.pt` for each observation. The reconstruction file contains original, initial and reconstructed images, client groups and optimisation histories. The scheme folder also keeps `observations.pt` and the separate `evaluation_targets.pt` for rerunning inversion.

Membership saves per-query scores in `scores.csv` and checkpoint measurements in `observations.npz`. Its `loss` values are negative cross-entropy; `clients` identifies their order, with `0` for the global model. Individual client losses and cosines are present only when the server observes them. Poisoning saves confusion matrices and a paired summary.

Procedure timings follow the native implementations: client training and encryption are averages across clients. Components may overlap and should not be summed as whole-round time; use `cost.csv` for that.

To rerun one saved GIA observation without training again:

```powershell
python -m attacks.gia.inverting `
  --attack-view results/gia/MNIST/trial_4010/fedavg/observations.pt `
  --evaluation-target results/gia/MNIST/trial_4010/fedavg/evaluation_targets.pt `
  --condition fedavg_server_aggregate `
  --output-directory results/gia_replay --iter 2400
```

## Acknowledgements

This project builds on:

- [SIHEFL](https://github.com/CameronLee02/server-initiated_HE_FL) — Lee et al.
- [GIA](https://github.com/1wrx1/GIA) — Guo et al.
- [DataPoisoning_FL](https://github.com/git-disl/DataPoisoning_FL) — Tolpegin et al.
- [FedMIA](https://github.com/Liar-Mask/FedMIA) — Zhu et al.
- [Opacus](https://github.com/pytorch/opacus) — differentially private training.
