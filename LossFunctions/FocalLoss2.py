import tensorflow as tf

def CustomLoss(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1)
    
    y_true = tf.cast(y_true, tf.int32)
    
    batch_indices = tf.range(tf.shape(y_pred)[0])
    
    indices = tf.stack([batch_indices, y_true], axis=1)
    
    correct_class_probs = tf.gather_nd(y_pred, indices)
    
    loss = -tf.math.log(correct_class_probs)*tf.square(1-correct_class_probs)
    
    return tf.reduce_mean(loss)
