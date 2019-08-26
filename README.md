## What is pycelle?
Pycelle is code that can be used to detect and quanify round bright regions in an image with a dark background. In the image, each stain/dot corresponds to a certain element that the user wants to characterize/quatify. For example, the stains/dots can be starts/galaxies in images captured by telescopes or droplets/micelles/cells in images from microscopes.

## How does pycelle work?

Pycelle converts the original image in a grayscale image. Then, gets the histogram of the grayscale image and detects the threshold (n) where the cumulative distribution is 75%.
To make the background more uniformely dark, the threshold n is applied to the image, converting all the values bellow n to 0 (black) in the grayscale image.

After that, the bright elements in the grayscale image (with threshold applied) are detected using a function that computes Laplacian of Gaussian images. See the [scikit image API](https://scikit-image.org/docs/dev/api skimage.feature.html#skimage.feature.blob_log) for more details. 

For each image, a csv file is created containing information about the position and radius of the bright elements.

Additionaly, another overview file, called analysis file, is created. Each line describes general features of an image: the number of elements, the percentage of bright regions in the image (%) and the mean radius of the elements. 

## Instructions

### Input 
For the input, pycelle expects JPG images inside a directory. 
A JSON file should go together with each JPG image. 
The JSON files contains the parameters used to detect the bright elements. 

- max_sigma_value: related to the maximum size of the elements
- min_sigma_value: related to the minimum size of the elements
- num_sigma_value: related to the minimum distance between the elements
- threshold_value: the smaller it is, the less the element needs to distinguish from the background be to be considered

### Run 

Pycelle is intended to be used with docker. You can choose one of these two methods do run the container. 

#### Method 1

Using this method it's possible to reproduce the results and apply the container to your own dataset. However, you can't change the parameters. 

1. Navigate to the directory where you want to work (e.g. $WORK):
```
$ cd $WORK
```

2. Run the container:
```
$ docker run -v $PWD/test:/data acpadua/pycelle:latest python3 /analyse_images.py
```

#### Method 2

The advantage of this second method is that you can edit the code if needed.

1. Navigate to the directory where you want to work (e.g. $WORK):
```
$ cd $WORK
```

2. Clone the repository from github to the current directory:
```
$ git clone https://github.com/acpadua/pycelle.git
```

3. Build the container, typing on your terminal:
```
$ cd pycelle
$ docker build -t pycelle:latest .
```

4. To run the code, type on your terminal:
```
$ docker run -v $PWD/test:/data pycelle:latest python3 /analyse_images.py
```

### Output

The results will appear at an output subdirectory inside the main directory. 
