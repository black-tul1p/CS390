##########################################################################
#      Name  : Divay Gupta      |     Email : gupta576@purdue.edu        #
##########################################################################

################################ Imports #################################
                                                                         #
import os                                                                #
import random                                                            #
import warnings                                                          #
import cv2 as cv                                                         #
import numpy as np                                                       #
import tensorflow as tf                                                  #
from tensorflow import keras                                             #
import tensorflow.keras.backend as K                                     #
from scipy.optimize import fmin_l_bfgs_b                                 #
from tensorflow.keras.applications import vgg19                          #
from tensorflow.keras.preprocessing.image import load_img, img_to_array  #
                                                                         #
##########################################################################

######################### Basic Initialization ###########################
                                                                         #
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.       #
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"                                 #
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                                                                         #
random.seed(1618)                                                        #
np.random.seed(1618)                                                     #
#tf.set_random_seed(1618)   # Uncomment for TF1.                         #
tf.random.set_seed(1618)                                                 #
                                                                         #
CONTENT_IMG_PATH = "./custom/fujairah.jpg"                               #
STYLE_IMG_PATH   = "./reference/pointillism.jpg"                         #
OUTPUT_IMG_PATH  = "./output"                                            #
                                                                         #
##########################################################################

########################### Global Constants #############################
                                                                         #
CONTENT_IMG_H = 500                                                      #
CONTENT_IMG_W = 800                                                      #
                                                                         #
STYLE_IMG_H = 500                                                        #
STYLE_IMG_W = 800                                                        #
                                                                         #
GRAD_DESC_STEP = 25                                                      #
GRAD_DESC_ITER = 1000                                                    #
                                                                         #
TRANSFER_ROUNDS = 3                                                      #
                                                                         #
##########################################################################

############################ TF Hyperparameters ##########################
                                                                         #
CONTENT_WEIGHT = 0.3    # Alpha weight.                                  #
STYLE_WEIGHT = 100.0    # Beta weight.                                   #
TOTAL_WEIGHT = 1.0                                                       #
                                                                         #
##########################################################################



############################ Helper Functions ############################

def deprocessImage(img):
    # Reshape input image
    img = img.copy()
    img = img.reshape((STYLE_IMG_H, STYLE_IMG_W, 3))

    # Reverse VGG19 transformations
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    # Revert image colorspace to RGB
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype("uint8")


def gramMatrix(x):
    # Check Image Data Format
    if (K.image_data_format() == "channels_first"):
        features = K.flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    # Compute and return gram matrix
    gram = K.dot(features, K.transpose(features))
    return gram

##########################################################################



#################### Loss Function Builder Functions #####################

def styleLoss(style, gen):
    return (K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / 
        (4.0 * np.square(style.shape[2]) * 
            np.square(STYLE_IMG_H * STYLE_IMG_W)))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    a = K.square(x[:, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1, :]
     - x[:, 1:, :STYLE_IMG_W - 1, :])
    b = K.square(x[:, :STYLE_IMG_H - 1, :STYLE_IMG_W - 1, :]
     - x[:, :STYLE_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

##########################################################################



########################### Pipeline Functions ###########################

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = cv.resize(img, dsize = (ih, iw))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")

    # Initialize image data tensors
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    
    # Define and load the VGG19 model
    model = vgg19.VGG19(include_top = False, weights = "imagenet", input_tensor = inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    
    # Initialize loss value
    loss = 0.0
    
    # Initialize layer names
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    
    # Compute content loss
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    contentGenOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, contentGenOutput)
    
    # Compute style loss
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        styleGenOutput = styleLayer[2, :, :, :]
        loss += STYLE_WEIGHT * styleLoss(styleOutput, styleGenOutput)

    # Setup gradients
    grads = K.gradients(loss, genTensor)

    # Setup outputs
    out = [loss, grads]

    # Initialize Loss Function
    kf = K.function([genTensor], out)
    def getLoss(x):
        x = x.reshape((1, STYLE_IMG_H, STYLE_IMG_W, 3))

        # Get loss
        loss, _ = kf([x])
        return (np.array(loss).flatten().astype("float64"))

    # Initialize Gradient Function
    def getGrads(x):
        x = x.reshape((1, STYLE_IMG_H, STYLE_IMG_W, 3))

        # Get gradient
        _, grad = kf([x])
        return (np.array(grad).flatten().astype("float64"))

    # Perform style transfer
    print("   Beginning transfer.")
    for i in range(1, TRANSFER_ROUNDS + 1):
        print("   Step %d." % i)

        # Perform gradient descent using fmin_l_bfgs_b.
        tData, tLoss, _ = fmin_l_bfgs_b(getLoss, tData.flatten(), getGrads, maxfun = GRAD_DESC_STEP, maxiter = GRAD_DESC_ITER)
        print("      Loss: %f." % tLoss)
        
        # Deprocess and save converted image
        img = deprocessImage(tData)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        saveFile = os.path.join(OUTPUT_IMG_PATH, f"conv_{i}.jpg")
        cv.imwrite(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)

    print("   Transfer complete.")

##########################################################################



################################## Main ##################################

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()

##########################################################################