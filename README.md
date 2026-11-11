# CorridorDiffuser

Core research code for **"CorridorDiffuser: Velocity-Aware Safe Corridor Generation for Motion Planning in Cluttered Environments"**.

This repository is a cleaned source snapshot of a completed research project. It is maintained for code access and project preservation; it is not a packaged software release.

This work has been accepted and published in *IEEE Transactions on Automation Science and Engineering (T-ASE)*. If you find this code or the associated methodology useful in your research, we would greatly appreciate it if you could cite our paper.

```bibtex
@ARTICLE{11745550,
  author={Liu, Xuekai and Hang, Peng and Zhang, Hao and Sun, Jian},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={CorridorDiffuser: Velocity-Aware Safe Corridor Generation for Motion Planning in Cluttered Environments}, 
  year={2026},
  volume={23},
  number={},
  pages={18861-18875},
  doi={10.1109/TASE.2026.3745653}}
```

## Contents

- `CorridorDiffuser/`: diffusion model, loss functions, data loader, training, and inference scripts.
- `dataset/`: map, path, trajectory, and velocity-aware safe-corridor generation.
- `sim/`: continuous-map, obstacle, and vehicle utilities used by the experiments.
- `planner/`: Hybrid A* and sampling-based downstream planning experiments.
- `center_path_traj/`: center-path and trajectory post-processing scripts.

Experiment-specific duplicate scripts, private working notes, manuscript files, generated figures, datasets, logs, model weights, and bulk outputs are intentionally excluded. Baseline copies whose upstream licensing still needs review are also kept out of this public repository.

## Environment

The original experiments were developed in Python with PyTorch. Install the main dependencies with:

```bash
pip install -r requirements.txt
```

Exact historical package versions were not preserved consistently, so `requirements.txt` lists required packages without claiming a fully locked environment.

## Data and checkpoints

The scripts use these repository-relative locations by default:

```text
data/maps/
data/targets/
data/trajectories/
checkpoints/best.pth
outputs/
```

Datasets and checkpoints are not stored in Git. A Hugging Face artifact archive is planned separately; until its identifier is added here, the checkpoint-dependent inference scripts require a locally supplied `checkpoints/best.pth`.

## Notes

- Training entry point: `CorridorDiffuser/main.py`.
- Inference entry point: `CorridorDiffuser/test.py`.
- Run the entry points from the repository root, for example `python CorridorDiffuser/main.py`; the historical scripts use local module imports and repository-relative data paths.
- The source is published as the final research snapshot and may require small environment-specific adjustments for a new machine.

## License

The repository is distributed under the MIT License. See `LICENSE`.



