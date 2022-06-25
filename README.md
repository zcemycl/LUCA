# LUCA [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![example workflow](https://github.com/zcemycl/LUCA/actions/workflows/python3-ubuntu.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/zcemycl/LUCA/badge.svg?branch=main)](https://coveralls.io/github/zcemycl/LUCA?branch=main)
LUCA is a collection of training procedures of DNN models. The repo is named after an evolutionary biology term -- Last Universal Common Ancestor, in an attempt to become the origin of any artificially intelligent creature -- Last Universal Common Artificial Intelligence.

## Set up Development Environment
1. Install packages in virtual environment.
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r install/py/requirements.txt
pip install -r install/py/requirements.dev.txt
```
2. Configure pre-commit hooks.
```
pre-commit install
```
3. Export PATH.
```
export PYTHONPATH=$PWD:$PYTHONPATH
```
