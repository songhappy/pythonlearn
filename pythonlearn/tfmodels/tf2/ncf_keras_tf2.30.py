import keras
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, Concatenate
import time
from bigdl.dataset import movielens
import numpy as np
import tensorflow as tf

# keras 2.4.3, tensorflow 2.3
def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    print(user_input)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(user_input)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                                   embeddings_initializer='uniform',
                                   embeddings_regularizer=None,
                                   input_length=1)(item_input)

    # Crucial to flatten an embedding vector!
    # user_embed = MLP_Embedding_User(user_input)
    user_latent = Flatten()(MLP_Embedding_User)
    item_latent = Flatten()(MLP_Embedding_Item)

    # The 0-th layer is the concatenation of embedding layers
    #vector = merge([user_latent, item_latent], mode='concat')
    vector = Concatenate()([user_latent, item_latent])
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx],  activation='relu',
                      name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(5, activation='softmax', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model

if __name__ == '__main__':
    epochs = 2
    batch_size = 1024
    movielens_data = movielens.get_id_ratings("/tmp/movielens/")
    print(movielens_data[:3])

    user_input = movielens_data[:,0]
    item_input = movielens_data[:,1]
    labels = movielens_data[:,2]
    user_input = np.array(list(map(lambda x: int(x.reshape([1])), user_input)))
    item_input = np.array(list(map(lambda x: int(x.reshape([1])), item_input)))
    def tozerobased(l):
        return l - 1
    labels = np.array(list(map(tozerobased, labels)))
    print(type(labels))
    print(np.array(user_input)[:10])
    print(np.array(item_input)[:10])
    print(labels[:10])
    import sys
    #sys.exit()
    num_users = max(user_input) + 1
    num_items = max(item_input) + 1
    model = get_model(num_users, num_items)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    for epoch in range(epochs):
        t1 = time.time()
        # Generate training instances

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         labels,  # labels
                         batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

    test = model.predict([np.array([10]), np.array([10])])
    print(test)

    filepath = "/Users/guoqiong/intelWork/git/tensorflow/tf-models/models/ncf23"

    tf.saved_model.save(model, filepath)
    loaded = tf.saved_model.load(filepath)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    test_input = [tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name='user_input'), tf.TensorSpec(shape=(None, 1), dtype=tf.int32, name='item_input')]
    test_input = [np.array([1]), np.array([1])]

    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(sess, ["serve"], filepath)
    #     graph = tf.get_default_graph()
    #     print(graph.get_operations())
    #     sess.run('myOutput:0',
    #              feed_dict={'myInput:0':test_input})