import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
import time
from bigdl.dataset import movielens
import numpy as np

# keras 1.2.2, tensorflow 1.15
def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2,
                                   name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] / 2,
                                   name='item_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu',
                      name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(5, activation='softmax', init='lecun_uniform', name='prediction')(vector)

    model = Model(input=[user_input, item_input],
                  output=prediction)

    return model

if __name__ == '__main__':
    epochs = 2
    batch_size = 1024
    movielens_data = movielens.get_id_ratings("/tmp/movielens/")
    print(movielens_data[:3])


    user_input = movielens_data[:,0]
    item_input = movielens_data[:,1]
    labels = movielens_data[:,2]
    # user_input = np.array(list(map(lambda x: x.reshape([1]), user_input)))
    # item_input = np.array(list(map(lambda x: x.reshape([1]), item_input)))
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
                         batch_size=batch_size, nb_epoch=1, verbose=1, shuffle=True)

        test = model.predict([np.array([1]), np.array([1])])
        print(test)

    # for i, layer in enumerate(model.layers):
    #     print(i)
    #     print(layer.get_weights())

    layer_user = model.get_layer("user_embedding")
    layer_item = model.get_layer("item_embedding")

    print(layer_user)
    print(layer_user.get_weights()[0])
    print(type(layer_user.get_weights()[0]))
    print(layer_user.get_weights()[0].shape)
