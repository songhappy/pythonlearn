import tensorflow as tf
sess = tf.get_default_session
import tensorflow.compat.v2 as tf

from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

model_path = '/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/tf_wnd'
imported = tf.saved_model.load(sess=sess, export_dir=model_path, tags=None,import_scope=None)
print(list(imported.signatures.keys()))
