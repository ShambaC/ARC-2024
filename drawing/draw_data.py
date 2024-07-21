# import tkinter
import json
import numpy as np
import cv2
# from PIL import Image, ImageTk

colorList = [[229,58,163],
             [146,18,49],
             [249,60,49],
             [255,133,27],
             [255,220,0],
             [79,204,48],
             [30,147,255],
             [135,216,241],
             [153,153,153],
             [0,0,0]]

def draw_grid(img, grid_shape, color=(0, 0, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

# window = tkinter.Tk()

# window.title = "Visualise"
# window.geometry("800x800")

jsonFile = open('../data/arc-agi_training_challenges.json')

trainingData = json.load(jsonFile)

for key in trainingData :
    print(key)
    trainData = trainingData[key]["train"]
    for numData in trainData :
        inputMat = numData['input']
        outputMat = numData['output']

        pixelSize = 20

        inputDimensionY = len(inputMat) * pixelSize
        inputDimensionX = len(inputMat[0]) * pixelSize

        outputDimensionY = len(outputMat) * pixelSize
        outputDimensionX = len(outputMat[0]) * pixelSize

        inputImg = np.full((inputDimensionY, inputDimensionX, 3), 255, dtype="uint8")
        outputImg = np.full((outputDimensionY, outputDimensionX, 3), 255, dtype="uint8")

        p = q = 0

        for i in range(0, inputDimensionY, pixelSize) :
            q = 0
            for j in range(0, inputDimensionX, pixelSize) :
                # print(f'i:{i},j:{j},p:{p},q:{q}')
                # print(inputMat[p][q])
                inputImg[i:i+pixelSize, j:j+pixelSize] = colorList[inputMat[p][q]]
                q += 1
            p += 1

        p = q = 0

        for i in range(0, outputDimensionY, pixelSize) :
            q = 0
            for j in range(0, outputDimensionX, pixelSize) :
                outputImg[i:i+pixelSize, j:j+pixelSize] = colorList[outputMat[p][q]]
                q += 1
            p += 1

        # Add grid to the image
        inputImg = draw_grid(inputImg, (len(inputMat), len(inputMat[0])))
        outputImg = draw_grid(outputImg, (len(outputMat), len(outputMat[0])))

        cv2.namedWindow(f'{key}_input', cv2.WINDOW_NORMAL)
        cv2.namedWindow(f'{key}_output', cv2.WINDOW_NORMAL)

        # manage window size based on image size
        # INPUT IMG
        if inputDimensionX > inputDimensionY :
            factor = 600 / inputDimensionX
            cv2.resizeWindow(f'{key}_input', 600, int(inputDimensionY * factor))
        elif inputDimensionX < inputDimensionY :
            factor = 600 / inputDimensionY
            cv2.resizeWindow(f'{key}_input', int(inputDimensionX * factor), 600)
        else :
            cv2.resizeWindow(f'{key}_input', 600, 600)

        # OUTPUT IMG
        if outputDimensionX > outputDimensionY :
            factor = 600 / outputDimensionX
            cv2.resizeWindow(f'{key}_output', 600, int(outputDimensionY * factor))
        elif outputDimensionX < outputDimensionY :
            factor = 600 / outputDimensionY
            cv2.resizeWindow(f'{key}_output', int(outputDimensionX * factor), 600)
        else :
            cv2.resizeWindow(f'{key}_output', 600, 600)

        while True :

            cv2.imshow(f'{key}_input', inputImg)
            cv2.imshow(f'{key}_output', outputImg)

            if cv2.waitKey(1) == ord('q') :
                break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()