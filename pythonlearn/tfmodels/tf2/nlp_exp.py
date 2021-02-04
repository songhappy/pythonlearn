from keras.datasets import imdb

import tensorflow as tf

sample_text = 'This is a\'s & sample sentence\n.'
x = tf.keras.preprocessing.text.text_to_word_sequence(
    sample_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
print(x)

