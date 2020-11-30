from typing import Callable, Optional

import tensorflow as tf

from tfrng.core import global_generator_context
from tfrng.generator import GeneratorGenerator
from tfrng.stateless import StatelessGenerator

Transform = Callable[[tf.data.Dataset], tf.data.Dataset]


def with_seed(seed: Optional[int] = None, size: Optional[int] = None) -> Transform:
    """
    Get a transform that zips a random seed along with dataset elements.

    Note the seeds for each element of the transformed dataset are generated
    pseudo-randomly and independently of the dataset.

    Example usage:
    ```python
    x = tf.data.Dataset.range(5)
    x = x.apply(tfrng.data.with_seed(0, size=3))
    print(x.element_spec)
    # (TensorSpec(shape=(3,), dtype=tf.int64, name=None),
    #  TensorSpec(shape=(), dtype=tf.int64, name=None))
    ```

    Args:
        seed: seed used in tf.data.experimental.RandomDataset which generates element
            seeds.
        size: size of each element seed.

    Returns:
        A transform that, when applied to a dataset with elements `element`, returns
        another dataset with elements `(element_seed, element)`.
    """

    def transform(dataset: tf.data.Dataset) -> tf.data.Dataset:
        random_dataset = tf.data.experimental.RandomDataset(seed)
        if size is not None:
            random_dataset = random_dataset.batch(size, drop_remainder=True)
        return tf.data.Dataset.zip((random_dataset, dataset))

    return transform


def stateless_map(
    map_func: Callable,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    deterministic: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Transform:
    """
    Similar to `tf.data.Dataset.map` but in a `StatelessGenerator` context.

    Note the resulting dataset will be deterministic if `deterministic` is True.
    Separate iterations over the dataset will yield the same results from tfrng ops.

    Example usage:
    ```python

    def map_func(i):
        return tf.cast(i, tf.float32) + tfrng.uniform(())


    map_kwargs = dict(
        num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

    dataset = tf.data.Dataset.range(8).apply(stateless_map(map_func, **map_kwargs))
    ```

    Arguments:
        map_func, num_parallel_calls, deterministic: see `tf.data.Dataset.map`
        seed: value used for `StatelessGenerator`

    Returns:
        A transform that can be applied to a dataset using `tf.data.Dataset.apply`.
    """

    def actual_map_func(element_seed, element):
        with global_generator_context(StatelessGenerator(element_seed)):
            if isinstance(element, tuple):
                return map_func(*element)
            return map_func(element)

    def transform(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.apply(with_seed(seed, 2)).map(
            actual_map_func,
            num_parallel_calls=num_parallel_calls,
            deterministic=deterministic,
        )

    return transform


def generator_map(
    map_func, num_parallel_calls: int = 1, rng: Optional[tf.random.Generator] = None
) -> Transform:
    """
    Similar to `tf.data.Dataset.map` but in a `GeneratorGenerator` context.

    Note the resulting dataset will not be deterministic if `num_parallel_calls > 1`.
    Separate iterations over the dataset will yield different results from tfrng ops.

    Arguments:
        map_func, num_parallel_calls: see `tf.data.Dataset.map`
        rng: Generator instance.

    Returns:
        A transform that can be applied to a dataset using `tf.data.Dataset.apply`.
    ```
    """

    def actual_map_func(*element):
        with global_generator_context(
            GeneratorGenerator(tf.random.get_global_generator() if rng is None else rng)
        ):
            return map_func(*element)

    def transform(dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(
            actual_map_func,
            num_parallel_calls=num_parallel_calls,
            deterministic=num_parallel_calls == 1,
        )

    return transform
