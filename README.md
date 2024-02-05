### Reverse Image Search

An example of building a reverse image search engine i.e. Content Based Image Retrieval

Code from the following blog post: 

https://www.cheeyeo.dev/machine-learning/torch/opencv/dinov2/image-retrieval/2024/01/02/image-retrieval/


### Dataset

Uses the [Caltech 256 dataset] for indexing and testing


### Using DINOV2 model

The model used to index the dataset features to create an embedding vector from is the DINOV2 model.

Based on experiements from using Convolutional models such as ResNet and ViT models such as CLIP, DINOV2 has the highest accuracy in terms of being able to learn image features specific to each category, based on results of manual search.

DINOV2 is trained using self-supervised learning without training labels so its able to generalize better to unseen datasets?

The above would require evaluation metrics to verify...

Both CLIP and ResNet models are unable to differentiate between object categories and sometimes retrieve images based on backgrounds or textures alone e.g matching an airplane with a helicopter with similar backgrounds; or returning image of rhino for query of image of elephant as they have similar textures...


### Setup

Run the weaviate DB via compose:
```
docker compose -f compose.yaml up
```

Create the schema first:
```
python create_schema.py
```

This creates a class of **Image** with 2 attributes:
* image, which stores a base64 encoding of the image
* filepath, which stores filepath of image


To index the images:
```
python upload_image_data.py
```

To start the webapp:
```
flask --app webapp/app run --debug
```

### Improvements / TODO

* Using PCA to perform dimensionality reduction to help improve extracted features to improve accuracy ?

* Do we need to perform individual object detection for each image?


### References

[Caltech 256 dataset]: https://data.caltech.edu/records/nyy15-4j048

[Weaviate guide]: https://weaviate.io/developers/weaviate/starter-guides/custom-vectors

[Weaviate Image Search tutorial]: https://weaviate.io/blog/how-to-build-an-image-search-application-with-weaviate

[Blog post on using Weaviate]: https://medium.com/@st3llasia/using-weaviate-to-find-similar-images-caddf32eaa3f

[Example of reverse image search app]: https://github.com/towhee-io/examples/tree/main/image/reverse_image_search

[DINO v2 image retrieval example]: https://www.kaggle.com/code/abdelkareem/dinov2-instance-retrieval

[Sample notebook of DINOV2 image retrieval]: https://github.com/roboflow/notebooks/blob/main/notebooks/dinov2-image-retrieval.ipynb


* [Weaviate guide]
* [Weaviate Image Search tutorial]
* [Blog post on using Weaviate]
* [DINO v2 image retrieval example]