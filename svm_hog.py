import cv2
import numpy as np

r = 25

winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)


test_svm = cv2.ml.SVM_load("pupil_svm_model.yml")

#hog.setSVMDetector(np.array(test_svm.getSupportVector(0)))



test_im = cv2.imread('05.bmp', 0)
im_res = test_im.copy()
h, w = np.shape(test_im)
#rects, weights = hog.detectMultiScale(test_im)
#print(rects, weights)
#exit()
#test_im = cv2.resize(test_im, (240, 320))
for x in range(r, w - r, 10):
    for y in range(r, h - r, 10):
        test_descriptor = hog.compute(test_im[y - r:y + r, x - r: x + r])
        test_data = []
        test_data.append(test_descriptor)
        trainData = np.float32(test_data).reshape(-1,np.shape(test_descriptor)[0])

        # Test on a held out test set
        testResponse = test_svm.predict(trainData)[1].ravel()

        if testResponse[0] == 1:
            cv2.rectangle(im_res, (x - r, y - r),  (x + r, y + r), 3)

cv2.imshow('result', im_res)
cv2.waitKey()


print("\nTest responses:")
print(testResponse)

exit()




f = open('data/data set new I.txt', 'r')
x = f.readlines()[1:-1]

y = []
for line in x:
    inner = [elt.strip() for elt in line.split(' ')]
    y.append(inner)

print(y)

hogdata = []
responses = []


i = 0
for e in y:
    im = cv2.imread('data/data set new I/0000' + e[1] + '.png', 0)
    x, y = int(int(e[2]) / 2), np.shape(im)[0] - int(int(e[3]) / 2)
    cropped = im[y - r:y + r, x - r:x + r]
    i += 1
    #if i > 2000:
    #    break
    if x + 3 * r >= np.shape(im)[1] or y + 3 * r >= np.shape(im)[0] or x - 3 * r < 0 or y - 3 * r < 0:
        continue
    cropped_false = im[y + r:y + 3 * r, x + r:x + 3 * r]
    descriptor = hog.compute(cropped)
    descriptor_false = hog.compute(cropped_false)

    cropped_false_1 = im[y - 3 * r:y - r, x - 3 * r:x - r]
    descriptor_false_1 = hog.compute(cropped_false_1)

    #print(np.shape(descriptor), np.shape(descriptor_false))

    hogdata.append(descriptor)
    hogdata.append(descriptor_false)
    hogdata.append(descriptor_false_1)
    #print(np.shape(hogdata))
    responses.append(1)
    responses.append(-1)
    responses.append(-1)
    #print(x, y)
    #cv2.imwrite('data/positives/' + e[1] + '_p.png', cropped)
    #cv2.imshow('0', cropped)
    #cv2.waitKey()

print(np.shape(hogdata))
print(len(responses))

trainData = np.float32(hogdata).reshape(-1,1296)
responses = np.array(responses)[:,np.newaxis]#np.repeat(np.arange(1) + 1,12167)[:,np.newaxis]

#responses = np.float32(np.repeat(np.arange(10),250)[:,np.newaxis])
# Set up SVM for OpenCV 3
svm = cv2.ml.SVM_create()
# Set SVM type
svm.setType(cv2.ml.SVM_C_SVC)
# Set SVM Kernel to Radial Basis Function (RBF)
svm.setKernel(cv2.ml.SVM_RBF)
# Set parameter C
svm.setC(12.5)
# Set parameter Gamma
svm.setGamma(0.50625)

# Train SVM on training data
svm.trainAuto(trainData, cv2.ml.ROW_SAMPLE, responses)

# Save trained model
svm.save("pupil_svm_model.yml");

# Test on a held out test set
testResponse = svm.predict(trainData)[1].ravel()

k = 0
err_count = 0
for a in testResponse:
    if k % 3 == 0 and a != 1:
        err_count += 1
    if k % 3 != 0 and a != -1:
        err_count += 1
    k += 1

print("\nTest responses:")
print(testResponse)

print("\nTotal errors: ", err_count)



#print(y)