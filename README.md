# Tensorflow Random Number Generation

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Tensorflow](https://tensorflow.org) has many different [random number generation implementations](https://www.tensorflow.org/api_docs/python/tf/random?hl=en).

* `tf.random` ops, e.g. `tf.random.uniform`
* `tf.random.Generator` methods
* `tf.random.stateless_*` ops, e.g. `tf.random.stateless_uniform`

All are intended to achieve similar things. This package presents a uniform interface and context manager to switch between these implementations.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
