import google.generativeai as genai
#GOOGLE_API_KEY=userdata.get('API')
genai.configure(api_key='API')
model = genai.GenerativeModel('gemini-pro')

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
response = model.generate_content("hello")

print(response.text)
'''
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True)

for chunk in response:
  print(chunk.text)
  print("_"*80)
'''
import PIL.Image

img = PIL.Image.open('image.jpg')
model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(img)

print(response.text)
