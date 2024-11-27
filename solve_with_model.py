from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"

#load the model and label binarizer
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

print("\nModel training info:")    
print("Model was trained on these characters:", sorted(list(lb.classes_)))
print("\nStarting prediction...\n")
    
model = load_model(MODEL_FILENAME)

#grab random CAPTCHA images to test against
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)

#keep track of correct predictions
correct = 0
total = 0

#loop over the image paths
for image_file in captcha_image_files:
    print("-" * 50)
    filename = os.path.basename(image_file)
    actual_text = filename.split('.')[0]
    print(f"Processing: {filename}")
    
    #load and preprocess the image
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    
    #add extra padding around the image
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    
    #threshold the image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    #find contours in the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    letter_image_regions = []
    
    #loop through each contour and extract the letter
    for contour in contours:
        #get rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        #compare width and height of contour to detect conjoined letters
        if w/h > 1.25:
            #split wide contours in half
            half_width = int(w/2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else: 
            #normal letter
            letter_image_regions.append((x, y, w, h))
            
    #if we found more or less than 4 letters, skip the image
    if len(letter_image_regions) != 4:
        continue
        
    #sort the detected letters from left to right
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    #create output image and predictions list
    output = cv2.merge([gray] * 3)
    predictions = []
    
    print("\nPredictions:")
    #loop over each letter box
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box
        
        #extract and preprocess the letter
        letter_image = gray[y-2:y+h+2, x-2:x+w+2]
        letter_image = resize_to_fit(letter_image, 20, 20)
        
        #prepare image for neural network
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)
        
        #ask neural network to make prediction
        prediction = model.predict(letter_image, verbose=0)
        confidence = np.max(prediction) * 100
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)
        
        print(f"  Predicted '{letter}' with {confidence:.1f}% confidence")
        
        #draw the prediction on the output image
        cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    #join all predictions into final text
    predicted_text = "".join(predictions)
    total += 1
    if predicted_text == actual_text:
        correct += 1
        
    #print results
    print(f"\nPredicted text: {predicted_text}")
    print(f"Actual text:    {actual_text}")
    print(f"Correct: {'Yes' if predicted_text == actual_text else 'No'}")
    
    #show the annotated image
    cv2.imshow("Output", output)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

#clean up and show final results
cv2.destroyAllWindows()
print("\nFinal Results:")
print(f"Correctly predicted: {correct}/{total} ({(correct/total)*100:.1f}%)")