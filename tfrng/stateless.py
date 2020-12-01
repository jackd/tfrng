import tensorflow as tf

from tfrng import core


# pylint: disable=signature-differs
class StatelessGenerator(core.Generator):
    """Generator implementation backed by `tf.random.stateless_*` ops."""

    def __init__(self, seed):
        seed = tf.convert_to_tensor(seed)
        if seed.shape != (2,):
            raise ValueError(f"`seed` must have shape (2,), got {tuple(seed.shape)}")
        self._seed = seed

    def _split_seed(self):
        seed = self._seed
        self._seed = tf.squeeze(
            tf.random.experimental.stateless_split(self._seed, 1), axis=0
        )
        return seed

    def uniform(self, shape, *args, **kwargs):
        return tf.random.stateless_uniform(shape, self._split_seed(), *args, **kwargs)

    def normal(self, shape, *args, **kwargs):
        return tf.random.stateless_normal(shape, self._split_seed(), *args, **kwargs)

    def binomial(self, shape, *args, **kwargs):
        if "dtype" in kwargs:
            assert "output_dtype" not in kwargs
            kwargs["output_dtype"] = kwargs.pop("dtype")
        return tf.random.stateless_binomial(shape, self._split_seed(), *args, **kwargs)

    def truncated_normal(self, shape, *args, **kwargs):
        return tf.random.stateless_truncated_normal(
            shape, self._split_seed(), *args, **kwargs
        )
