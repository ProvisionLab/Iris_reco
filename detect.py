import cv2
import numpy as np
import sys
import os


def isInsideTheCircle(x0, y0, r0, x, y):
    if (x0 - x) * (x0 - x) + (y0 - y) * (y0 - y) <= r0 * r0:
        return True
    return False


def sumOverCircle(img_, x0, y0, r0):
    total = 0
    for x in range(x0 - r0, x0 + r0):
        for y in range(y0 - r0, y0 + r0):
            if isInsideTheCircle(x0, y0, r0, x, y):
                total += img_.item(y, x)

        #print(cv2.sqrt(r0*r0 - (x - x0) * (x - x0)))
        '''
        y1 = int(y0 + cv2.sqrt(r0*r0 - (x - x0) * (x - x0))[0][0])
        y2 = int(y0 - cv2.sqrt(r0*r0 - (x - x0) * (x - x0))[0][0])
        total += edges.item(y1, x)
        total += edges.item(y2, x)
        total += edges.item(y2 + 1, x + 1)
        total += edges.item(y1 + 1, x + 1)
        total += edges.item(y2 + 1, x - 1)
        total += edges.item(y1 + 1, x - 1)
        total += edges.item(y2 - 1, x + 1)
        total += edges.item(y1 - 1, x + 1)
        total += edges.item(y2 - 1, x - 1)
        total += edges.item(y1 - 1, x - 1)
        total += edges.item(y2, x + 1)
        total += edges.item(y1, x + 1)
        total += edges.item(y2, x - 1)
        total += edges.item(y1, x - 1)
        total += edges.item(y2 + 1, x)
        total += edges.item(y1 + 1, x)
        total += edges.item(y2 - 1, x)
        total += edges.item(y1 - 1, x)
        '''

    total_normalized = total / (np.pi * r0 * r0)
    return total_normalized


if __name__ == "__main__":
    directory = sys.argv[1]
    processing_sz = (80, 60)
    for img_path in os.listdir(directory):
        #print(img_path)
        img = cv2.imread(directory + img_path, 0)
        img = cv2.resize(img, processing_sz)
        #img = cv2.blur(img, (, 9))

        # img = cv2.bilateralFilter(img, 9, 75, 75)
        (height, width) = img.shape

        mask = cv2.imread(directory + img_path, 0)
        mask = cv2.resize(mask, processing_sz)
        #mask = cv2.blur(mask, (9, 9))
        for x in range(0, width):
            for y in range(0, height):
                mask[y][x] = 0

        for x in range(int(width / 4), int(width * 3 / 4)):
            for y in range(int(height / 4), int(height * 3 / 4)):
                if 240 < img.item(y, x):
                    mask[y][x] = 255

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

        cv2.imwrite("results/" + img_path + "_inpaint_mask.jpg", mask)

        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        cv2.imwrite("results/" + img_path + "_inpainted.jpg", img)


        sorted_img = np.sort(img, axis=None)
        #print(sorted_img)
        ethalon = sorted_img[int(height * width / 100)]
        print(ethalon)

        for x in range(0, width):
            for y in range(0, height):
                if img.item(y, x) > ethalon:
                    img[y][x] = 255
                else:
                    img[y][x] = 0


        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
        # sure background area
        #img = opening
        img = cv2.dilate(opening, kernel, iterations=1)

        cv2.imwrite("results/" + img_path + "_mask.jpg", img)

        print(height, width)
        max_radius = 18#int(width / 6)
        min_radius = 1#int(width / 25)
        step = 2#int(min_radius / 20 + 1)

        #print(min_radius)
        #print(max_radius)
        #print(step)
    
        max_sum = 256

        top_x = 0; top_y = 0; top_r = 0
        for r_ in range(min_radius, max_radius, step):
            for y_ in range(int(height / 4), int(height * 3 / 4), step):
                for x_ in range(0, int(width * 3 / 4), step):
                    if x_ - r_ <= 0 or x_ + r_ >= width:
                        continue
                    if y_ - r_ <= 0 or y_ + r_ >= height:
                        continue
                    current = sumOverCircle(img, x_, y_, r_)
                    #print(current)
                    if current < 100.0 and r_ > top_r:
                        max_sum = current
                        top_x = x_
                        top_y = y_
                        top_r = r_

        print(top_x, top_y)
        print(top_r)
    #grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #image[top_y][top_x] = [0, 0, 255]
        image = cv2.imread(directory + img_path)
        image = cv2.resize(image, processing_sz)
        cv2.circle(image, (top_x, top_y), 1, [0, 0, 255], 2)
        #print("/results/" + img_path + "_res.jpg")
        cv2.imwrite("results/" + img_path + "_res.jpg", image)
    #cv2.waitKey()


    """
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    image = cv2.imread(img_path)
    image = cv2.resize(image, (80, 60))
    markers = cv2.watershed(image, markers)
    #image[markers == -1] = [0, 0, 255]

    #cv2.imshow('bin', image)

    norm_image = img
    cv2.normalize(img, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #cv2.imshow('img', norm_image)

    edges = cv2.Laplacian(img, cv2.CV_64F)

    #for x in range(0, width):
    #    for y in range(0, height):
    #        if edges.item(y, x) < 20:
    #            edges[y][x] = 0

    #print(markers[-1])
    #cv2.imshow('result', image)
    #cv2.imwrite('res.jpg', image)
    """
