import abc
import contextlib
import inspect

import tensorflow as tf


class Generator(abc.ABC):
    """Interface for random number generator implementations."""

    @abc.abstractmethod
    def uniform(self, shape, minval=0, maxval=None, dtype=tf.dtypes.float32, name=None):
        """Outputs random values from a uniform distribution."""

    @abc.abstractmethod
    def normal(self, shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None):
        """Outputs random values from a normal distribution."""

    @abc.abstractmethod
    def binomial(self, shape, counts, probs, dtype=tf.dtypes.int32, name=None):
        """Outputs random values from a binomial distribution."""

    @abc.abstractmethod
    def truncated_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None
    ):
        """Outputs random values from a truncated normal distribution."""

    def uniform_full_int(self, shape, dtype=tf.dtypes.uint64, name=None):
        """
        Uniform distribution on an integer type's entire range.

        This method is the same as setting minval and maxval to None in the uniform
        method.
        """
        return self.uniform(shape, maxval=dtype.max, name=name)

    def shuffle(self, value, name=None):
        """Shuffle along the leading dimension of value."""
        with tf.name_scope(name or "shuffle"):
            if hasattr(value, "nrows"):
                size = value.nrows()
            elif hasattr(value, "dense_shape"):
                size = value.dense_shape[0]
            else:
                size = tf.shape(value)[0]
            r = self.uniform((size,))
            perm = tf.argsort(r)
            return tf.gather(value, perm, axis=0)


# pylint: disable=unexpected-keyword-arg
class StatefulGenerator(Generator):
    """
    Generator that uses stateful `tf.random` ops, e.g. `tf.random.uniform`.

    The returned ops themselves have state.

    The default global generator is an instance of this class. Different instances are
    equivalent.
    """

    def uniform(self, shape, minval=0, maxval=None, dtype=tf.dtypes.float32, name=None):
        """Outputs random values from a uniform distribution."""
        return tf.random.uniform(
            shape=shape, minval=minval, maxval=maxval, dtype=dtype, name=name
        )

    def normal(self, shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None):
        """Outputs random values from a normal distribution."""
        return tf.random.normal(
            shape=shape, mean=mean, stddev=stddev, dtype=dtype, name=name
        )

    def truncated_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None
    ):
        """Outputs random values from a truncated normal distribution."""
        return tf.random.truncated_normal(
            shape=shape, mean=mean, stddev=stddev, dtype=dtype, name=name
        )

    def binomial(self, shape, counts, probs, dtype=tf.dtypes.int32, name=None):
        """Outputs random values from a binomial distribution."""
        return tf.random.stateless_binomial(
            shape=shape,
            counts=counts,
            probs=probs,
            output_dtype=dtype,
            seed=self.uniform_full_int((2,)),
        )
        # return tf.random.binomial(
        #     shape=shape, counts=counts, probs=probs, dtype=dtype, name=name
        # )

    def shuffle(self, value, name=None):
        """Shuffle along the leading dimension of value."""
        return tf.random.shuffle(value, name=name)


# pylint: enable=unexpected-keyword-arg

_GLOBAL_GENERATOR = StatefulGenerator()


def set_global_generator(generator: Generator):
    global _GLOBAL_GENERATOR  # pylint: disable=global-statement
    _GLOBAL_GENERATOR = generator


def get_global_generator():
    return _GLOBAL_GENERATOR


@contextlib.contextmanager
def global_generator_context(generator: Generator):
    old_generator = get_global_generator()
    try:
        set_global_generator(generator)
        yield
    finally:
        set_global_generator(old_generator)


def _export_global_method(name: str):
    """
    Export f(*args, **kwargs) = getattr(get_global_generator(), name)(*args, **kwargs)).

    Exported function has same name, signature (without `self`) and docs.
    """
    method = getattr(Generator, name)

    def fn(*args, **kwargs):
        gen = get_global_generator()
        return getattr(gen, name)(*args, **kwargs)

    sig = inspect.signature(method)
    params = tuple(sig.parameters.values())[1:]
    fn.__signature__ = sig.replace(parameters=params)
    fn.__doc__ = method.__doc__
    fn.__name__ = name
    return fn


uniform = _export_global_method("uniform")
uniform_full_int = _export_global_method("uniform_full_int")
normal = _export_global_method("normal")
truncated_normal = _export_global_method("truncated_normal")
binomial = _export_global_method("binomial")
shuffle = _export_global_method("shuffle")
