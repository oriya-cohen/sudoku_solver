import pytesseract
from PIL import Image
import requests
import io

response = requests.get('https://i.stack.imgur.com/oAAXR.png')
text = pytesseract.image_to_string(Image.open(io.BytesIO(response.content)), lang='eng',
                    config='--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')

print(text)