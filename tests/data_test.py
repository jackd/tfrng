import functools

import numpy as np
import tensorflow as tf

import tfrng

base = tf.data.Dataset.range(4, output_type=tf.float32)


def transform(x, scale, shift):
    return (x + shift) * scale


def tfrng_map_func(x):
    scale = tfrng.normal((), stddev=0.1, mean=1.0)
    shift = tfrng.uniform(())
    return transform(x, scale, shift)


def naive_map_func(x):
    scale = tf.random.normal((), stddev=0.1, mean=1.0)
    shift = tf.random.uniform(())
    return transform(x, scale, shift)


def evaluate(dataset):
    return list(dataset.as_numpy_iterator())


# pylint: disable=no-self-use
class TestDataMaps(tf.test.TestCase):
    def test_naive(self):
        tf.random.set_seed(0)
        actual = evaluate(base.map(tfrng_map_func))
        tf.random.set_seed(0)
        expected = evaluate(base.map(naive_map_func))
        np.testing.assert_equal(actual, expected)

    def test_generator_map(self):
        rng = tf.random.Generator.from_seed(0)
        actual = evaluate(base.apply(tfrng.data.generator_map(tfrng_map_func, rng=rng)))

        def rng_map_func(x, rng):
            scale = rng.normal((), stddev=0.1, mean=1.0)
            shift = rng.uniform(())
            return transform(x, scale, shift)

        rng = tf.random.Generator.from_seed(0)
        expected = evaluate(base.map(functools.partial(rng_map_func, rng=rng)))
        np.testing.assert_equal(actual, expected)

    def test_stateless_map(self):
        actual = evaluate(
            base.apply(
                tfrng.data.stateless_map(tfrng_map_func, num_parallel_calls=4, seed=0)
            )
        )

        # stateless`
        def stateless_map_func(seed, x):
            scale = tf.random.stateless_normal((), stddev=0.1, mean=1.0, seed=seed)
            seed1 = tf.squeeze(tf.random.experimental.stateless_split(seed, 1), axis=0)
            shift = tf.random.stateless_uniform((), seed=seed1)
            return transform(x, scale, shift)

        random_dataset = tf.data.experimental.RandomDataset(seed=0).batch(2)
        expected = evaluate(
            tf.data.Dataset.zip((random_dataset, base)).map(
                stateless_map_func, num_parallel_calls=4
            )
        )
        np.testing.assert_equal(actual, expected)
