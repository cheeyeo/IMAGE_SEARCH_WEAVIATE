import io
import json
import os
import base64
import argparse
import weaviate
from PIL import Image
from models.resenet50 import ResNet50Vectorizer
from models.clipmodel import ClipImageEmbed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='Image to query for')
    args, _ = ap.parse_known_args()

    WEAVIATE_URL = os.getenv('WEAVIATE_URL')

    if not WEAVIATE_URL:
        WEAVIATE_URL = 'http://localhost:8080'

    client = weaviate.Client(WEAVIATE_URL)

    max_distance = 0.18
    # model = ResNet50Vectorizer()
    model = ClipImageEmbed()
    query_vector = {
        'vector': model.embed(args.image),
        'distance': max_distance
    }

    res = (
        client.query.get(
            'Image', ['filepath']
        )
        .with_near_vector(query_vector)
        .with_limit(5)
        .with_additional(['distance'])
        .do()
    )

    print(json.dumps(res, indent=2))


