# TODO: set up colab?
import functools

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


# naive
print("naive_map_func with map")
tf.random.set_seed(0)
dataset = base.map(naive_map_func)
print(list(dataset.as_numpy_iterator()))

print("tfrng_map_func with map")
tf.random.set_seed(0)
dataset = base.map(tfrng_map_func)
print(list(dataset.as_numpy_iterator()))


# rng
def rng_map_func(x, rng):
    scale = rng.normal((), stddev=0.1, mean=1.0)
    shift = rng.uniform(())
    return transform(x, scale, shift)


rng = tf.random.Generator.from_seed(0)
rng.reset_from_seed(0)
dataset = base.map(functools.partial(rng_map_func, rng=rng))
print("rng_map_func with map")
print(list(dataset.as_numpy_iterator()))
rng = tf.random.Generator.from_seed(0)
dataset = base.apply(tfrng.data.generator_map(tfrng_map_func, rng=rng))
print("tfrng_map_func with generator_map")
print(list(dataset.as_numpy_iterator()))

# stateless
def stateless_map_func(seed, x):
    scale = tf.random.stateless_normal((), stddev=0.1, mean=1.0, seed=seed)
    seed1 = tf.squeeze(tf.random.experimental.stateless_split(seed, 1), axis=0)
    shift = tf.random.stateless_uniform((), seed=seed1)
    return transform(x, scale, shift)


random_dataset = tf.data.experimental.RandomDataset(seed=0).batch(2)
dataset = tf.data.Dataset.zip((random_dataset, base)).map(
    stateless_map_func, num_parallel_calls=4
)
print("stateless_map_func with map")
print(list(dataset.as_numpy_iterator()))

print("tfrng_map_func with stateless_map")
dataset = base.apply(
    tfrng.data.stateless_map(tfrng_map_func, seed=0, num_parallel_calls=4)
)
print(list(dataset.as_numpy_iterator()))
