import imutils
import cv2

def resize_to_fit(image, width, height):
    #helper function to resize an image to fit within a given size
    #image: image to resize
    #width: desired width in pixels
    #height: desired height in pixels
    #returns the resized image
    
    #gets dimensions of the image, the intializes the padding values
    (h, w) = image.shape[:2]
    
    #if width greater than height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)
    #height greater than width so resize along height    
    else:
        image = imutils.resize(image, height=height)
    
    #determine padding values for the width and height to obtain target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    
    #pad the image then apply one more resizing for potential rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    
    #return preprocessed image
    return image