import tensorflow as tf

def carpole_net_target(num_actions):
    block_input = tf.placeholder(tf.float32, [None, 1])
    B0 = tf.layers.dense(
        inputs=block_input,
        units=4,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
        bias_initializer=tf.constant_initializer(0.1),
    )
    value = tf.layers.dense(
        inputs=B0,
        units= num_actions,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
        bias_initializer=tf.constant_initializer(0.1)
    )

    return  value, block_input