import PIL.Image
import os
import google.generativeai as genai
import cv2

os.chdir(r"C:\Users\rayso_sq9ff\Downloads")
genai.configure(api_key='AIzaSyCnIphJLd9FdZq_cXiSItw-2yvENi8-03k')

img = PIL.Image.open(r"C:\Users\rayso_sq9ff\Downloads\jpegmini_optimized\IMG_5803.jpg")
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)

print(response.text)
input="there is only three bottle in the images,Which bottle is the hand pointing? from left to right, give me the name of the photo, there is only three photo, each bottle's distance from point to straight line are 1.40cm 2.5cm 3.60cm"
response = model.generate_content([
                                      input,
                                      img], stream=True)
response.resolve()

print(response.text)
