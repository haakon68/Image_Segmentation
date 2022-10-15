from importlib.resources import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

seeds = []
# Calculate the difference between the pixel value of the seed point and its field
def getGrayDiff(gray, current_seed, tmp_seed):
    return abs(int(gray[current_seed[0], current_seed[1]]) - int(gray[tmp_seed[0], tmp_seed[1]]))

# Region growing algorithm
def regional_growth(gray, seeds, threshold, num_connects):
    # Four neighboors
    if num_connects == 4:
        connects = [(-1, 0), (0, 1), (0, -1), (1, 0)] 
    # Eight fields
    if num_connects == 8:
        connects = [(-1, -1), (0, -1), (1, -1),
                    (-1, 0), (1, 0),
                    (-1, 1), (0, 1), (1, 1)]
    seedMark  = np.zeros((gray.shape))
    height, width = gray.shape
    seedque = deque()
    label = 255
    seedque.extend(seeds)

    while seedque:
        # Queues are first in first out. So delete it from the left
        current_seed = seedque.popleft()
        seedMark[current_seed[0], current_seed[1]] = label 
        for i in range(num_connects):
            tmpX = current_seed[0] + connects[i][0]
            tmpY = current_seed[1] + connects[i][1]
            
            # Dealing with ther border situations
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue

            grayDiff = getGrayDiff(gray, current_seed, (tmpX, tmpY))
            if grayDiff < threshold and seedMark[tmpX, tmpY] != label:
                seedque.append((tmpX, tmpY))
                seedMark[tmpX, tmpY] = label
    return seedMark 

# Interaction function 
def event_mouse(event, x, y, flags, param):
    # Left click mouse
    if event ==  cv2.EVENT_LBUTTONDOWN:
        # Add seeds
        seeds.append((y, x))
        # Draw solid dots
        cv2. circle(img, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)

def region_grow(path_img, img, threshold, num_connects):
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', event_mouse)
    cv2.imshow('img', img)
    
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    CT = cv2.imread(path_img)
    seedMark = np.uint8(regional_growth(cv2.cvtColor(CT, cv2.COLOR_BGR2GRAY), seeds, threshold, num_connects))
    cv2.imshow('seedMark', seedMark)
    cv2.waitKey(0)
    
    plt.figure(figsize=(12, 3))
    plt.subplot(131), plt.imshow(cv2.cvtColor(CT, cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title(f'$input\_image$')
    plt.subplot(132), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off'), plt.title(f'$seeds\_image$')
    plt.subplot(133), plt.imshow(seedMark, cmap='gray', vmin = 0, vmax = 255)
    plt.axis('off'), plt.title(f'$segmented\_image$')
    plt.tight_layout()
    plt.show()


# ==================================================================================================


# split 
def Division_Judge(img, h0, w0, h, w) :
    area = img[h0 : h0 + h, w0 : w0 + w]
    mean = np.mean(area)
    std = np.std(area, ddof = 1)

    total_points = 0
    operated_points = 0

    for row in range(area.shape[0]) :
        for col in range(area.shape[1]) :
            if (area[row][col] - mean) < 2 * std :
                operated_points += 1
            total_points += 1

    if operated_points / total_points >= 0.95 :
        return True
    else :
        return False

def Merge(img, h0, w0, h, w) :
    # area = img[h0 : h0 + h, w0 : w0 + w]
    # _, thresh = cv.threshold(area, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
    # img[h0 : h0 + h, w0 : w0 + w] = thresh
    for row in range(h0, h0 + h) :
        for col in range(w0, w0 + w) :
            if img[row, col] > 100 and img[row, col] < 200:
                img[row, col] = 0
            else :
                img[row, col] = 255

def Recursion(img, h0, w0, h, w) :
    # If the splitting conditions are met, continue to split 
    if not Division_Judge(img, h0, w0, h, w) and min(h, w) > 5 :
        # Recursion continues to determine whether it can continue to split 
        # Top left square 
        Division_Judge(img, h0, w0, int(h0 / 2), int(w0 / 2))
        # Upper right square 
        Division_Judge(img, h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
        # Lower left square 
        Division_Judge(img, h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
        # Lower right square 
        Division_Judge(img, h0 + int(h0 / 2), w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
    else :
        # Merge 
        Merge(img, h0, w0, h, w)

def Division_Merge_Segmented(img_path, show=True) :
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(img_gray, bins = 256)
    print(f' Five-pointed star 、 The ellipse 、 background 、 The pixel values of pentagons are ：'
          f'{"、".join("%s" % pixel for pixel in np.unique(img_gray))}')

    segemented_img = img_gray.copy()
    Recursion(segemented_img, 0, 0, segemented_img.shape[0], segemented_img.shape[1])

    if show:
        plt.figure(figsize=(12, 4))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off'), plt.title(f'$input\_image$')
        plt.subplot(132), plt.imshow(img_gray, cmap='gray', vmin = 0, vmax = 255)
        plt.axis('off'), plt.title(f'$gray\_image$')
        plt.subplot(133), plt.imshow(segemented_img, cmap='gray')
        plt.axis('off'), plt.title(f'$segmented\_image$')
        plt.tight_layout()
        plt.show()

    return img, img_gray, segemented_img


    
if __name__ == '__main__':

    option = input('Enter option: ')

    try:
        option = int(option)
    except Exception:
        print('Error!!!!')  

    if option == 1:
        path_img = input('Path of image: ')
        threshold = int(input('Threshold: '))
        num_connects = int(input('Number of connects: '))
        img = cv2.imread(path_img)
        region_grow(path_img, img, threshold, num_connects)


    elif option == 2:
        path_img = input('Path of image: ')
        Division_Merge_Segmented(path_img)

    else:
        quit()

