# Installation

```bash
git clone https://github.com/keejkrej/cell-filter.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-filter.git
cd cell-filter
pip install -e .
```

# Usage

First analyze, then extract

## Pattern (test if patterns are properly recognized)

```bash
python -m cell_filter.cli.pattern
--patterns ./20220525_patterns_end.nd2
--cells ./20220525_MBAMB231.nd2
--view 0
--output ./pattern.png
```

## Analyze

```bash
python -m cell_filter.cli.analyze
--patterns ./20220525_patterns_end.nd2
--cells ./20220525_MBAMB231.nd2
--nuclei-channel 1
--cyto-channel 0
--output ./output/analysis
--range 0:10
--wanted 3
```

## Extract

```bash
python -m cell_filter.cli.extract
--patterns ./20220525_patterns_end.nd2
--cells ./20220525_MBAMB231.nd2
--time-series ./output/analysis
--output ./output
--min-frames 20
--nuclei-channel 1
--cyto-channel 0
```

# Code Structure

## CLI

- [`analyze.py`](src/cell_filter/cli/analyze.py)
- [`extract.py`](src/cell_filter/cli/extract.py)
- [`pattern.py`](src/cell_filter/cli/pattern.py)

## Core

- [`analyze.py`](src/cell_filter/core/analyze.py)
- [`generate.py`](src/cell_filter/core/generate.py)
- [`count.py`](src/cell_filter/core/count.py)
- [`extract.py`](src/cell_filter/core/extract.py)
- [`pattern.py`](src/cell_filter/core/pattern.py)
