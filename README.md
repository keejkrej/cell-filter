# Installation

```bash
git clone https://github.com/keejkrej/cell-counter.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-counter.git
cd cell-counter
pip install -e .
```

# Usage

First analyze, then extract

## Analyze

```bash
python -m cell_counter.cli.analyze \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--output /path/to/output/analysis \
--start-view 0 \
--end-view 10 \
--wanted 3
```

## Extract

```bash
python -m cell_counter.cli.extract \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--time-series /path/to/output/analysis \
--output /path/to/output \
--min-frames 20
```

# Code Structure

## CLI

- [`analyze.py`](src/cell_counter/cli/analyze.py)
- [`extract.py`](src/cell_counter/cli/extract.py)
- [`info.py`](src/cell_counter/cli/info.py)

## Core

- [`Analyzer.py`](src/cell_counter/core/Analyzer.py)
- [`CellGenerator.py`](src/cell_counter/core/CellGenerator.py)
- [`CellposeCounter.py`](src/cell_counter/core/CellposeCounter.py)
- [`Extractor.py`](src/cell_counter/core/Extractor.py)
- [`InfoDisplayer.py`](src/cell_counter/core/InfoDisplayer.py)
