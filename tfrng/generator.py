from typing import Optional

import tensorflow as tf

from tfrng import core

# pylint: disable=signature-differs


class GeneratorGenerator(core.Generator, tf.Module):
    """Generator implementation backed by `tf.random.Generator` methods."""

    def __init__(self, rng: tf.random.Generator, name: Optional[str] = None):
        assert isinstance(rng, tf.random.Generator)
        self._rng = rng
        tf.Module.__init__(self, name=name)

    def uniform(self, *args, **kwargs):
        return self._rng.uniform(*args, **kwargs)

    def normal(self, *args, **kwargs):
        return self._rng.normal(*args, **kwargs)

    def truncated_normal(self, *args, **kwargs):
        return self._rng.truncated_normal(*args, **kwargs)

    def binomial(self, *args, **kwargs):
        return self._rng.binomial(*args, **kwargs)

    def uniform_full_int(self, *args, **kwargs):
        return self._rng.uniform_full_int(*args, **kwargs)


# pylint: enable=signature-differs
