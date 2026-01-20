import tensorflow as tf

def CustomLoss(y_true, y_pred):

    batch_indices = tf.range(tf.shape(y_pred)[0])
    
    # Convert y_true to int32 (required for tf.gather)
    y_true = tf.cast(y_true, tf.int32)

    # Combine indices so we can pick the right log probability for each sample
    indices = tf.stack([batch_indices, y_true], axis=1)
    
    # Gather the predicted probability of the correct class
    correct_class_probs = tf.gather_nd(y_pred, indices)

    return tf.reduce_mean((y_true - correct_class_probs) ** 2)
