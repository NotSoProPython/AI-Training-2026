"""

This is the custom loss function that the model uses to train

"""
import tensorflow as tf

def CustomLoss(y_true, y_pred): # YTrue is the one-hot encoded (0,1,0,0,...) vector with 1 on the correct number, YPred is the model's output values
    # Ensure predictions are normalized probabilities
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1)  # Prevent log(0)
    
    # Convert y_true to int32 (required for tf.gather)
    y_true = tf.cast(y_true, tf.int32)
    
    # Get batch indices for each sample
    batch_indices = tf.range(tf.shape(y_pred)[0])
    
    # Combine indices so we can pick the right log probability for each sample
    indices = tf.stack([batch_indices, y_true], axis=1)
    
    # Gather the predicted probability of the correct class
    correct_class_probs = tf.gather_nd(y_pred, indices)
    
    # Compute loss: -log(p_correct)
    loss = -tf.math.log(correct_class_probs)
    
    # Return the mean loss
    return tf.reduce_mean(loss)