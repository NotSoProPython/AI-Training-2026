print("Initializing tensorflow...")

from tensorflow.keras.models import Sequential # type:ignore
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Softmax # type:ignore 
import tensorflow_datasets as tfds
from LossFunctions.Debug import CustomLoss
import numpy as np
import random
from tensorflow.python.distribute import input_lib

# Temporary fix for missing attribute because tensorflow is a buggy piece of garbage
if not hasattr(input_lib, "DistributedDatasetInterface"):
    input_lib.DistributedDatasetInterface = input_lib.DistributedDatasetSpec

def TrainModel(id=None, weights=None, epochs=5, strong=False, translates=False):

    # Defining Model Architecture
    if strong:
        LeNetModel = Sequential([
            Conv2D(filters=10, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
            AveragePooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=200, activation='relu'),
            Dense(units=120, activation='relu'),
            Dense(units=84, activation='relu'),
            Dense(units=30, activation='relu'),
            Dense(units=10, activation='softmax')
            ])
    else:
        LeNetModel = Sequential([
            Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)),
            AveragePooling2D(pool_size=(2, 2)),
            Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
            AveragePooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=120, activation='relu'),
            Dense(units=84, activation='relu'),
            Dense(units=10, activation='softmax')
            ])

    # THIS IS WHERE YOU CHANGE THE LOSS FUNCTION
    LeNetModel.compile(optimizer='adam', loss=CustomLoss, metrics=['accuracy'])
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Skip training with premade weights
    if weights:
        LeNetModel.load_weights(weights)
        return LeNetModel

    print("Loading up MNIST data...")

    Train, Test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    print("Converting training set to arrays...")

    TrainingImages = []
    TrainingAnswers = []
    for image, label in tfds.as_numpy(Train):
        TrainingImages.append(image)
        TrainingAnswers.append(label)
    TrainingImages = np.array(TrainingImages)
    TrainingAnswers = np.array(TrainingAnswers)

    print("Converting testing set to arrays...")

    TestingImages = []
    TestingAnswers = []
    for image, label in tfds.as_numpy(Test):
        TestingImages.append(image)
        TestingAnswers.append(label)
    TestingImages = np.array(TestingImages)
    TestingAnswers = np.array(TestingAnswers)

    print("Preprocessing images...")

    # Now do the same preprocessing
    if not translates:
        TrainingImages = np.pad(TrainingImages, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    else:
        TempTrainingAnswers = TrainingAnswers
        for _ in range(24):
            TempTrainingAnswers = np.append(TempTrainingAnswers, TrainingAnswers, axis=0)
        TrainingAnswers = TempTrainingAnswers
        TrainingImages = np.squeeze(TrainingImages)
        Translates = []
        Counter = 0
        for i in range(0,5):
            for j in range(0,5):
                Counter += 1
                print(f"Processing batch {Counter}/25")
                Translates.append(np.pad(TrainingImages, ((0,0),(i,4-i),(j,4-j)), 'constant'))
        TrainingImages = np.concatenate(Translates)
        RandomPermutation = np.random.permutation(TrainingImages.shape[0])
        TrainingAnswers = TrainingAnswers[RandomPermutation]
        TrainingImages = TrainingImages[RandomPermutation]
       
    TestingImages = np.pad(TestingImages, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    TrainingImages = TrainingImages.astype('float32') / 255.0
    TestingImages = TestingImages.astype('float32') / 255.0

    TrainingImages = np.expand_dims(TrainingImages, -1)
    TestingImages = np.expand_dims(TestingImages, -1)

    print("Reviewing model architecture...")

    # Display the model summary to check the architecture
    LeNetModel.summary()

    print("Training model...")

    # Train
    LeNetModel.fit(TrainingImages, TrainingAnswers, epochs=epochs, batch_size=128, validation_data=(TestingImages, TestingAnswers))

    print("Saving model weights...")
    if not id:
        LeNetModel.save_weights("Convolutional Neural Network\\Previous Weights\\Weights" + str(random.randint(0,999999)) + ".weights.h5")
    else:
        LeNetModel.save_weights("Convolutional Neural Network\\Previous Weights\\" + str(id) + ".weights.h5")
    
    return LeNetModel

def SampleSet():

    Train, Test = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    TestingImages = []
    for image, label in tfds.as_numpy(Test):
        TestingImages.append(image)

    TestingImages = np.array(TestingImages)
    TestingImages = np.pad(TestingImages, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    TestingImages = TestingImages.astype('float32') / 255.0
    TestingImages = np.expand_dims(TestingImages, -1)

    return TestingImages