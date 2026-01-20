# Note: run with python main.py > log.txt

from train import TrainModel, SampleSet
import pygame
import numpy as np
import random

# pygame
pygame.init()
ScreenDim = (460,320)
Screen = pygame.display.set_mode(ScreenDim)
Clock = pygame.time.Clock()
Running = True
Font = pygame.font.SysFont("Arial", 12)

Input = np.full((32,32), 0, dtype=float)
Model = TrainModel(id="MSE3",epochs=3,strong=False,translates=False) # Weights3 is promising
#Model = TrainModel(strong=False, weights="Convolutional Neural Network\Previous Weights\Weights100.weights.h5")
Samples = SampleSet()
MousePressed = 0
MousePos = (-1, -1)
DrawStrength = 3
Prediction = -1
Confidence = 0
Drew = 0

# Helper Function
def SanitizeImage(image):
    if image.shape == (28,28):
        image = np.pad(image, ((2,2),(2,2)), 'constant')
    image = image.astype('float')
    image = np.squeeze(image)
    image = image.clip(max=1)
    image = np.expand_dims(image, axis=-1)  # (32,32,1)
    image = np.expand_dims(image, axis=0)   # (1,32,32,1)
    return image

# Main Loop
while Running:

    # Event Handler
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            MousePressed = True

        elif event.type == pygame.MOUSEBUTTONUP:
            MousePos = (-1, -1)
            MousePressed = False

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            Input = np.full((32,32), 0, dtype=float)
            Prediction = -1
            Confidence = 0
        
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
            Input = Samples[random.randint(0,9999)]
            Output = Model.predict(SanitizeImage(Input))
            Prediction = np.argmax(Output, axis=1)[0]
            Confidence = np.max(Output)
    
    Screen.fill((0,0,0))

    # Detecting the drawing
    CurrentPos = pygame.mouse.get_pos()
    Drew = max(Drew - 1, 0) # Fast vs Slow prediction
    if CurrentPos[0] < 321 and MousePos != CurrentPos and MousePressed:
        Drew = 5
        MousePos = CurrentPos
        XPos, YPos = CurrentPos
        XTile = (int(0.1*XPos), 1 - (XPos % 10)/10)
        YTile = (int(0.1*YPos), 1 - (YPos % 10)/10)
        Input[YTile[0],XTile[0]] += DrawStrength * XTile[1] * YTile[1]
        Input[YTile[0] + 1,XTile[0]] += DrawStrength * XTile[1] * (1 - YTile[1])
        Input[YTile[0],XTile[0] + 1] += DrawStrength * (1 - XTile[1]) * YTile[1]
        Input[YTile[0] + 1,XTile[0] + 1] += DrawStrength * (1 - XTile[1]) * (1 - YTile[1])
        Input = Input.clip(max=1)

    # Rendering the Input
    for i in range(31):
        for j in range(31):
            pygame.draw.rect(Screen, (Input[i,j] * 255, Input[i,j] * 255, Input[i,j] * 255), (10 * j, 10 * i, 10, 10))

    # Processing the Input
    if not Drew:
        Drew = 3600
        Output = Model.predict(SanitizeImage(Input))
        Prediction = np.argmax(Output, axis=1)[0]
        Confidence = np.max(Output)

    # Rendering the Output
    if Confidence > 0.1:
        FontSurface = Font.render(f"Detected: {Prediction}", True, (255,255,255))
        Screen.blit(FontSurface, FontSurface.get_rect(center=(390, 120)))
        FontSurface = Font.render(f"Confidence: {Confidence*100:.2f}%", True, (255,255,255))
        Screen.blit(FontSurface, FontSurface.get_rect(center=(390, 200)))
    elif Confidence != 0:
        FontSurface = Font.render(f"Unsure of digit", True, (255,255,255))
        Screen.blit(FontSurface, FontSurface.get_rect(center=(390, 160)))
    else:
        FontSurface = Font.render(f"Draw a digit", True, (255,255,255))
        Screen.blit(FontSurface, FontSurface.get_rect(center=(390, 160)))

    pygame.display.flip()

    Clock.tick(60)

pygame.quit()