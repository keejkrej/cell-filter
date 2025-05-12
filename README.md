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

## Info

```bash
python -m cell_filter.cli.info \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--view 0 \
--output /path/to/output/analysis \
```

## Analyze

```bash
python -m cell_filter.cli.analyze \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--nuclei-channel 1 \
--cyto-channel 0 \
--output /path/to/output/analysis \
--range 0:10 \
--wanted 3 \
```

## Extract

```bash
python -m cell_filter.cli.extract \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--time-series /path/to/output/analysis \
--output /path/to/output \
--min-frames 20 \
--nuclei-channel 1 \
--cyto-channel 0
```

# Code Structure

## CLI

- [`analyze.py`](src/cell_filter/cli/analyze.py)
- [`extract.py`](src/cell_filter/cli/extract.py)
- [`info.py`](src/cell_filter/cli/info.py)

## Core

- [`Analyzer.py`](src/cell_filter/core/Analyzer.py)
- [`CellGenerator.py`](src/cell_filter/core/CellGenerator.py)
- [`CellposeCounter.py`](src/cell_filter/core/CellposeCounter.py)
- [`Extractor.py`](src/cell_filter/core/Extractor.py)
- [`InfoDisplayer.py`](src/cell_filter/core/InfoDisplayer.py)
