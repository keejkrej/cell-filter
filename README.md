# Installation

```bash
git clone https://github.com/keejkrej/cell-filter.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-filter.git
cd cell-filter
pip install -e .
```

See [PROJECT.md](PROJECT.md) for a high-level project summary.

# Usage

- You can either copy [scripts/pattern.py](scripts/pattern.py), [scripts/analyze.py](scripts/analyze.py), and [scripts/extract.py](scripts/extract.py) to your working directory and run them directly after editing parameters, or use the provided package CLI entrypoints.

- Recommended: use the package entrypoints so you can keep configuration in your working directory and run via the installed package. Examples:

```bash
# show CLI help
python -m cell_filter.cli.pattern --help
python -m cell_filter.cli.analysis --help
python -m cell_filter.cli.extract --help

# run the same sequence as before (edit args as needed)
python -m cell_filter.cli.pattern --patterns data/20250806_patterns_after.nd2 --cells data/20250806_MDCK_timelapse_crop_fov0004.nd2
python -m cell_filter.cli.analysis --patterns data/20250806_patterns_after.nd2 --cells data/20250806_MDCK_timelapse_crop_fov0004.nd2
python -m cell_filter.cli.extract --patterns data/20250806_patterns_after.nd2 --cells data/20250806_MDCK_timelapse_crop_fov0004.nd2
```
