# Installation

```bash
git clone https://github.com/keejkrej/cell-filter.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-filter.git
cd cell-filter
pip install -e .
```

# Usage

## Analyze

```bash
python -m cell_filter.cli.analyze \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--output /path/to/output/analysis \
--start-view 0 \
--end-view 10 \
--wanted 3
```

## Extract

```bash
python -m cell_filter.cli.extract \
--patterns /path/to/20220525_patterns_end.nd2 \
--cells /path/to/20220525_MBAMB231.nd2 \
--time-series /path/to/output/analysis \
--output /path/to/output \
--min-frames 20
```