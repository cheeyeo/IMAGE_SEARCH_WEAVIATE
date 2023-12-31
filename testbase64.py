import base64
import io
from PIL import Image


if __name__ == '__main__':
    img = Image.open('test.jpg')
    print(type(img))
    format = img.format
    img = img.convert('RGB')
    print(type(img))

    img_bytes = io.BytesIO()
    img.save(img_bytes, format=format)
    img_bytes = img_bytes.getvalue()
    print(type(img_bytes))

    res = base64.b64encode(img_bytes)
    # print(res)
