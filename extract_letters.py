import os
import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"

#only set to True when debugging is needed
DEBUG_VISUALIZATION = False

#grab all image file paths
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
total_images = len(captcha_image_files)
counts = {}

print(f"\nTotal CAPTCHA images found: {total_images}")
processed = 0
skipped = 0

#loop over each image
for (i, captcha_image_file) in enumerate(captcha_image_files):
    #show progress for every 100 images processed
    if i % 100 == 0:
        print(f"Processing image {i}/{total_images} ({(i/total_images)*100:.1f}%)")
    
    #gets base filename as text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    
    #load image
    image = cv2.imread(captcha_image_file)
    
    #skip if image failed to load
    if image is None:
        skipped += 1
        continue
    
    #convert to grayscale and preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    #threshold the image and apply morphological operations
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    #show threshold image if debugging
    if DEBUG_VISUALIZATION:
        cv2.imshow("Threshold", thresh)
        cv2.waitKey(1)
    
    #find contours (continuous blobs of pixels) in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_image_regions = []
    
    #loop through each contour and extract the letter
    for contour in contours:
        #get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        #filter out contours that are too small
        if w * h < 100:
            continue
            
        #compare width and height of contour to detect conjoined letters
        if w/h > 1.25:
            #this contour is too wide to be a single letter
            #split it into half to create two letter regions
            half_width = int(w/2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            #normal letter by itself
            letter_image_regions.append((x, y, w, h))
            
    #don't have exactly 4 letters, skip the image
    if len(letter_image_regions) != 4:
        skipped += 1
        continue
        
    #sort detected letter images based on x coordinate to ensure left-to-right processing
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    #process and save each letter
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        #get coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        
        #extract letter from original image with a 2-pixel margin around edge
        letter_image = gray[y-2:y+h+2, x-2:x+w+2]
        
        #get folder to save letter image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        
        #create output directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        #write the letter image to a file with incrementing count
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)
        
        #increment the count for this letter
        counts[letter_text] = count + 1
    
    processed += 1

#print final statistics
print("\nFinal Statistics:")
print(f"Total images processed: {processed}")
print(f"Total images skipped: {skipped}")
print("Characters extracted:", sorted(counts.keys()))
print("Number of samples per character:", {k: v-1 for k, v in counts.items()})

#cleanup visualization windows if debugging was enabled
if DEBUG_VISUALIZATION:
    cv2.destroyAllWindows()