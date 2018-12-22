import tensorflow as tf
import pdb
def _phase_shift(I, r, batch_size = 10):
    # Helper function with main phase shift operation
#    pdb.set_trace()
    _, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (batch_size, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    if batch_size == 1:
        X = tf.expand_dims( X, 0 )
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    if batch_size == 1:
        X = tf.concat([x for x in X], 2 )
    else:
        X = tf.concat([tf.squeeze(x) for x in X], 2)  #
    out =  tf.reshape(X, (batch_size, a*r, b*r, 1))
    if batch_size == 1:
        out = tf.transpose( out, (0,2,1,3)  )
    return out

def PS(X, r, n_channel = 8, batch_size = 10):
  # Main OP that you can arbitrarily use in you tensorflow code
    Xc = tf.split(X, n_channel, -1 )
    X = tf.concat([_phase_shift(x, r, batch_size) for x in Xc], 3)
    return X
