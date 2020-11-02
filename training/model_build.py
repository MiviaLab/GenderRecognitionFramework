import keras
import sys


def senet_model_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    print("Building senet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeeze-excite-network')
    from keras_squeeze_excite_network.se_resnet import SEResNet
    m1 = SEResNet(weights=weights, input_shape=input_shape, include_top=True, pooling='avg',weight_decay=0)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, use_bias=True, activation='softmax', name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_224_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    print("Building mobilenet v2", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 1.0, include_top=True, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def vgg16_keras_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    # Alpha version
    print("Building vgg16", input_shape, "- num_classes", num_classes, "- weights", weights)
    from keras.applications.vgg16 import VGG16

    # # Uncomment these lines and check the loss
    # input_tensor = keras.layers.Input(shape=input_shape)
    # from keras.applications.vgg16 import preprocess_input
    # input_tensor = keras.layers.Lambda(preprocess_input, arguments={'mode': 'tf'})(input_tensor)
    # m1 = VGG16(include_top=True, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None)

    m1 = VGG16(include_top=True, weights=weights, input_tensor=None, input_shape=input_shape, pooling=None)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def densenet_121_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet", lpf_size=1):
    print("Building densenet121bc", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras_vggface')
    from keras_vggface.densenet import DenseNet121
    m1 = DenseNet121(include_top=True, input_shape=input_shape, weights=weights, pooling='avg', lpf_size=lpf_size)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_64_build(input_shape=(64, 64, 3), num_classes=2):
    print("Building mobilenet 64", input_shape, "- num_classes", num_classes)
    from scratch_models.mobile_net_v2_keras import MobileBioNetv2
    m1 = MobileBioNetv2(input_shape=input_shape, width_multiplier=0.5)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def mobilenet_96_build(input_shape=(96,96,3), num_classes=2, weights="imagenet"):
    print("Building mobilenet 96", input_shape, "- num_classes", num_classes, "- weights", weights)
    m1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape, 0.75, include_top=True, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def xception_build(input_shape=(299,299,3), num_classes=2, weights="imagenet", lpf_size=1):
    print("Building xception", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras_vggface')
    from keras_vggface.xception import Xception
    m1 = Xception(include_top=False, input_shape=input_shape, weights=weights, pooling='avg', lpf_size=lpf_size) #emulate include_top through pooling avg
    features = m1.layers[-1].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.models.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def squeezenet_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    print("Building squeezenet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-squeezenet')
    from keras_squeezenet import SqueezeNet
    m1 = SqueezeNet(input_shape=input_shape, weights=weights, include_top=True)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def shufflenet_224_build(input_shape=(224, 224, 3), num_classes=2, weights="imagenet"):
    print("Building shufflenet", input_shape, "- num_classes", num_classes, "- weights", weights)
    sys.path.append('keras-shufflenetV2')
    from shufflenetv2 import ShuffleNetV2
    m1 = ShuffleNetV2(input_shape=input_shape, classes=num_classes, include_top=True, scale_factor=1.0, weights=weights)
    features = m1.layers[-2].output
    x = keras.layers.Dense(num_classes, activation='softmax', use_bias=True, name='Logits')(features)
    model = keras.Model(m1.input, x)
    for l in model.layers: l.trainable = True
    return model, features


def vggface_custom_build(input_shape, num_classes=2, weights="vggface2", net="vgg16", lpf_size=1):
    sys.path.append('keras_vggface')
    from keras_vggface.vggface import VGGFace
    return VGGFace(model=net, weights=weights, input_shape=input_shape, classes=num_classes, lpf_size=lpf_size)
    



    



