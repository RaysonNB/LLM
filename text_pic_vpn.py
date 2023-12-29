import PIL.Image
import google.generativeai as genai
genai.configure(api_key='AIzaSyCnIphJLd9FdZq_cXiSItw-2yvENi8-03k')
img = PIL.Image.open(r'C:\Users\rayso_sq9ff\Downloads\archive (2)\Animals\Lion\test.jpg')
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)

print(response.text)

response = model.generate_content(["Write a short, engaging blog post based on this picture. It should include a description of these animal features. In chinese", img], stream=True)
response.resolve()

print(response.text)
