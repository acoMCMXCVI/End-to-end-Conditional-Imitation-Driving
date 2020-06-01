""" (2016). End to End Learning for Self-Driving Cars. NVIDIA.
Bojarski, Mariusz & Testa, Davide & Dworakowski, Daniel & Firner,
Bernhard & Flepp, Beat & Goyal, Prasoon & Jackel, Larry & Monfort,
Mathew & Muller, Urs & Zhang, Jiakai & Zhang, Xin & Zhao, Jake & Zieba,
Karol.
Links:
    - [Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
"""

from keras.models import Model
from keras.layers import Lambda, Dropout, ELU
from keras.layers import Conv2D, Flatten, Dense, Input


### Keras model architecture definition
# If under-fitting, more conv layers or increase epochs
# If over-fitting, less conv or dense layers, use dropout/BN for regularization, collect more data


def nvidia_model(width, height, depth):

    input = Input(shape=(height,width,depth))

	# Normalizing data in range of -1 to 1 and zero-centering data
    normalized = Lambda(lambda x: x / 127.5-1.0)(input)


    # Layer 1: 5x5 Conv + ELU + 2x2 MaxPool
    X = Conv2D(24, (5,5), strides=(2, 2), padding='valid', activation='relu')(normalized)  #treba  padding='valid'


	# Layer 2: 5x5 Conv + ELU + 2x2 MaxPool
    X = Conv2D(36, (5,5), strides=(2, 2), padding='valid', activation='relu')(X)


	# Layer 3: 5x5 Conv + ELU + 2x2 MaxPool
    X = Conv2D(48, (5,5), strides=(2, 2), padding='valid', activation='relu')(X)


	# Layer 4: 3x3 Conv + ELU + 2x2 MaxPool
    X = Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu')(X)

	# Layer 5: 3x3 Conv + ELU + 2x2 MaxPool + Dropout(drop_prob=0.5)
    X = Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu')(X)
    X = Dropout(0.5)(X)

	# Layers 6-8: Fully connected + ELU activation
    X = Flatten()(X)
    X = Dense(100, activation = 'relu')(X)
    X = Dense(50, activation = 'relu')(X)
    X = Dense(10, activation = 'relu')(X)
    output = Dense(1)(X)

    model = Model(inputs=input, outputs=output)

    model.summary()

    return model
