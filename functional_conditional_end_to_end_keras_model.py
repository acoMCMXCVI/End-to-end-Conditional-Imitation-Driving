""" (2016). End to End Learning for Self-Driving Cars. NVIDIA.
Bojarski, Mariusz & Testa, Davide & Dworakowski, Daniel & Firner,
Bernhard & Flepp, Beat & Goyal, Prasoon & Jackel, Larry & Monfort,
Mathew & Muller, Urs & Zhang, Jiakai & Zhang, Xin & Zhao, Jake & Zieba,
Karol.
Links:
    - [Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
"""

from keras.models import Model
from keras.layers import Lambda, Dropout, ELU, Concatenate
from keras.layers import Conv2D, Flatten, Dense, Input



def nvidia_model(width, height, depth, control):

    img_input = Input(shape=(height,width,depth))

	# Normalizing data in range of -1 to 1 and zero-centering data
    img_normalized = Lambda(lambda x: x / 127.5-1.0)(img_input)
    # Layer 1: 5x5 Conv + ELU + 2x2 MaxPool
    img_X = Conv2D(24, (5,5), strides=(2, 2), padding='valid', activation='relu')(img_normalized)  #treba  padding='valid'
	# Layer 2: 5x5 Conv + ELU + 2x2 MaxPool
    img_X = Conv2D(36, (5,5), strides=(2, 2), padding='valid', activation='relu')(img_X)
	# Layer 3: 5x5 Conv + ELU + 2x2 MaxPool
    img_X = Conv2D(48, (5,5), strides=(2, 2), padding='valid', activation='relu')(img_X)
	# Layer 4: 3x3 Conv + ELU + 2x2 MaxPool
    img_X = Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu')(img_X)
	# Layer 5: 3x3 Conv + ELU + 2x2 MaxPool + Dropout(drop_prob=0.5)
    img_X = Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu')(img_X)
    img_X = Dropout(0.5)(img_X)
	# Layers 6-8: Fully connected + ELU activation
    img_X = Flatten()(img_X)


    # control input
    control_input = Input(shape=(control,))

    control_X = Dense(50, activation = 'relu')(control_input)
    control_X = Dense(512, activation = 'relu')(control_X)

    # Concatenate control input and img input
    conc = Concatenate(axis=-1)([img_X, control_X])


    X = Dense(500, activation = 'relu')(conc)
    X = Dense(100, activation = 'relu')(X)
    X = Dense(50, activation = 'relu')(X)
    X = Dense(10, activation = 'relu')(X)
    output = Dense(1)(X)

    model = Model(inputs=[img_input, control_input], outputs=output)

    model.summary()

    return model
