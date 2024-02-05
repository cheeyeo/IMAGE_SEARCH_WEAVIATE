# Creates vectors if images and upload to weaviate db

from pathlib import Path
import os
import io
import base64
import weaviate
from PIL import Image
from tqdm import tqdm
from models.dinov2 import DinoV2Embed


def setup_batch(client):
    """
    Prepare batching config for Weaviate
    """

    client.batch.configure(
        batch_size=100,
        dynamic=True,
        timeout_retries=3,
        callback=None
    )


def delete_images(client):
    """
    Remove all images from vector db
    """
    with client.batch as batch:
        batch.delete_objects(
            class_name='Image',
            where={
                'operator': 'NotEqual',
                'path': ['filepath'],
                'valueString': 'x'
            },
            output='verbose'
        )



def img_to_base64(img_path):
    """
    img_content is PIL.Image ?
    """

    img = Image.open(img_path)
    img_format = img.format
    img = img.convert('RGB') # PIL.Image.Image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=img_format)
    img_bytes = img_bytes.getvalue()

    return base64.b64encode(img_bytes).decode('utf-8')


def import_data(client, source_path):
    """
    Process all images and upload its vector into db
    """
    model = DinoV2Embed()

    with client.batch as batch:
        for img_path in Path(source_path).rglob('**/*.jpg'):
            if img_path.is_file():
                # print(f'IMG PATH: {img_path}')
                tqdm.write(f'IMG PATH: {img_path}')

                img_vector = model.embed(img_path)
                img_base64 = img_to_base64(img_path)

                data_properties = {
                    'image': img_base64,
                    'filepath': str(img_path)
                }

                batch.add_data_object(data_properties, 'Image', vector=img_vector)


if __name__ == '__main__':
    WEAVIATE_URL = os.getenv('WEAVIATE_URL')
    if not WEAVIATE_URL:
        WEAVIATE_URL = 'http://localhost:8080'

    client = weaviate.Client(WEAVIATE_URL)

    setup_batch(client)

    delete_images(client)

    # Looks for subdir inside dataset directory
    p = Path('dataset')
    for child in tqdm(p.iterdir(), disable=None):
        tqdm.write(f'DIR: {child}')
        import_data(client, child)
