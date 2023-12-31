import os
from pathlib import Path
import json
from flask import Flask, render_template
from flask import request, flash, redirect, url_for
import weaviate
import sys
# add the submodules to $PATH
# sys.path[0] is the current file's path
sys.path.append(sys.path[0] + '/..')
from models.resenet50 import ResNet50Vectorizer
from models.clipmodel import ClipImageEmbed


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    file = request.files['file']
    # Upload file to tmp location and pass path as query to vector DB
    tmp_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(tmp_filename)

    client = weaviate.Client(WEAVIATE_URL)
    max_distance = 0.20
    # model = ResNet50Vectorizer()

    model = ClipImageEmbed()
    query_vector = {
        'vector': model.embed(tmp_filename),
        'distance': max_distance
    }

    res = (
        client.query.get(
            'Image', ['filepath', 'image']
        )
        .with_near_vector(query_vector)
        .with_limit(10)
        .with_additional(['distance'])
        .do()
    )

    print(json.dumps(res, indent=2))

    images = res['data']['Get']['Image']
    for img in images:
        img['filepath'] = '/'.join(Path(img['filepath']).parts[1:])

    return render_template('results.html', content=images)