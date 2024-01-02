import cv2
import time
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

#processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
#model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to("cuda")
print("Ready!")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from the camera. Please check your camera connection.")
        break
    keyCode = cv2.waitKey(1)
    if keyCode & 0xFF == ord('q'):
        break
    elif keyCode & 0xFF == ord(' '):
        # Convert the frame to PIL image format
        raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        question = input("Enter your question: ")
        start = time.time()
        inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True), "Time used: ", str(time.time() - start))

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the program


# Release the capture
cap.release()
cv2.destroyAllWindows()
