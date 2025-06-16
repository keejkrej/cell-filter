# Installation

```bash
git clone https://github.com/keejkrej/cell-filter.git
(or)
git clone https://gitlab.physik.uni-muenchen.de/LDAP_ls-raedler/cell-filter.git
cd cell-filter
pip install -e .
```

# Usage

- copy [scripts/pattern.py](scripts/pattern.py), [scripts/analyze.py](scripts/analyze.py), and [scripts/extract.py](scripts/extract.py) to your working directory,
- edit the parameters in the scripts,
- run the scripts in the following order:

```bash
python pattern.py
python analyze.py
python extract.py
```
