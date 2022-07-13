import requests
import io
import os
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

imagedir = "D:\\Pictures\\Skribbl Dataset\\images"

upload_url = "".join([
    "https://api.roboflow.com/dataset/cringw/upload",
    "?api_key=API_KEY"  # Almost forgot to remove my API key lol
])

for subdir, dirs, files in os.walk(imagedir):
    for file in files:
        image = Image.open(os.path.join(subdir, file)).convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")

        m = MultipartEncoder(fields={'file': (file, buffered.getvalue(), "image/jpeg")})
        r = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type})

        # Output result
        print(r.json())
