import os
import weaviate


WEAVIATE_URL = os.getenv('WEAVIATE_URL')
if not WEAVIATE_URL:
    WEAVIATE_URL = 'http://localhost:8080'


client = weaviate.Client(WEAVIATE_URL)

class_obj = {
    "class": "Image",
    "vectorizer": "none",
    "properties": [
        {
            "name": "image",
            "dataType": ["blob"],
            "description": "image"
        },
        {
            "name": "filepath",
            "dataType": ["string"],
            "description": "filepath of image"
        }
    ]
}

client.schema.create_class(class_obj)


