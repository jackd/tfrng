import numpy as np
import tensorflow as tf

import tfrng  # pylint: disable=import-error


def map_func(i):
    return tf.cast(i, tf.float32) + tfrng.uniform(())


map_kwargs = dict(num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)
base = tf.data.Dataset.range(5)

# incorrect deterministic implementation
dataset = base.map(map_func, **map_kwargs)
print("Using `Dataset.map`:")
print([el.numpy() for el in dataset])
print([el.numpy() for el in dataset])


# deterministic implementation
print("Using `stateless_map`")
dataset = base.apply(tfrng.data.stateless_map(map_func, seed=0, **map_kwargs))

print([el.numpy() for el in dataset])
print([el.numpy() for el in dataset])
print("Repeated with `stateless_map`")
print(np.reshape([el.numpy() for el in dataset.repeat(2)], (2, -1)))


print([el.numpy() for el in dataset])
print([el.numpy() for el in dataset])
print("Repeated with `stateless_map`")
print(np.reshape([el.numpy() for el in dataset.repeat(2)], (2, -1)))


# deterministic implementation, different across iterations
print("Using `generator_map` and num_parallel_calls=1:")
tf.random.get_global_generator().reset_from_seed(0)
dataset = base.apply(tfrng.data.generator_map(map_func, num_parallel_calls=1))

print([el.numpy() for el in dataset])
print([el.numpy() for el in dataset])
print("Repeated with `generator_map`")
print(np.reshape([el.numpy() for el in dataset.repeat(2)], (2, -1)))

print("Using `interleave` / `stateless_map` / `with_seed`:")
dataset = (
    tf.data.Dataset.from_tensors(base)  # a dataset with a single dataset element
    .repeat(2)  # now a dataset of datasets
    .apply(tfrng.data.with_seed(0, None))  # elements are (seed, dataset)
    .interleave(
        lambda element_seed, ds: ds.apply(
            tfrng.data.stateless_map(map_func, seed=element_seed, **map_kwargs),
        ),
        num_parallel_calls=2,
        cycle_length=2,
    )
)
print(np.reshape([el.numpy() for el in dataset], (-1, 2)))
