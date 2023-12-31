### Reverse Image Search

An example of building a reverse image search engine i.e. Content Based Image Retrieval


Based on CHAP 4 of book:

'Practical Deep Learning for Cloud, Mobile, Edge'



### Dataset

Uses the `caltech101` dataset for indexing and testing




### Using CLIP for feature extraction

Model is inside models/clip.py

The issue with the ResNet model extractor is that its not very accurate

Since CLIP has zero-shot object detection we can use the image encoder which is a ViT to encode image features...


ResNet model, when trained on entire dataset of all categories, it started to return images which have similar background features but not similar content i.e. when matching a fighter jet, it returns images of helicopters with similar background...

The same issue appears with the CLIP model but only for certain categories such as searching for elephants it might return some rhino images...

The image vector is normalized in the CLIP model compared to the ResNet model; not sure if that has an effect...


### Improvements

* Using PCA to perform dimensionality reduction to help improve extracted features to improve accuracy ??

* How do we detect each object in image automatically?






### Weaviate

https://weaviate.io/developers/weaviate/starter-guides/custom-vectors

https://weaviate.io/blog/how-to-build-an-image-search-application-with-weaviate

https://medium.com/@st3llasia/using-weaviate-to-find-similar-images-caddf32eaa3f



### Steps

* Feature extraction via Clip model pretrained.

* Create db schema to store image vectors and other attrs such as filename and image blob

* Import image data

* Run query


```
python create_schema.py

python upload_image_data.py

```

To run webapp:
```
flask --app webapp/app run --debug
```