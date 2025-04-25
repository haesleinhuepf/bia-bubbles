# BIA-bubbles: Test Your Bioimage Analysis Skills! 🔬🖥️🕹️

BIA-bubbles is an interactive game where you can put your bioimage analysis knowledge by solving bio-image analysis puzzles! 

![](https://github.com/haesleinhuepf/bia-bubbles/raw/main/docs/teaser.gif)

## The Challenge

Can you process biological images to achieve the perfect result? Each level presents you with:
- A starting microscopy image
- A target result you need to achieve
- A set of image processing tools at your disposal
- Real-time quality metrics to evaluate your solution

Your mission is to build the correct image analysis pipeline by:
1. Selecting the right processing operations
2. Applying them in the correct order
3. Achieving high accuracy compared to the target result

## Installation

```
pip install bia-bubbles
```

## How it works

You can start bia-bubbles from the terminal like this:
```
bia-bubbles
```

Everything works by clicking or touching on a touch-screen. You can click on an image and select image processing steps. Use the mouse wheel to zoom in or out. You can also push around images to arrange them. To get rid of intermediate results, just push them against the playground border. Once you found a solution that's good enough, you move on to the next level.

![](https://github.com/haesleinhuepf/bia-bubbles/raw/main/docs/bia-bubbles-demo.gif)
[Download high resolution video](https://github.com/haesleinhuepf/bia-bubbles/raw/main/docs/bia-bubbles-demo.mp4)

Ready to test your bioimage analysis skills? Jump in and see if you can solve all the challenges! Whether you're a beginner learning the fundamentals or an expert honing your skills, BIA-bubbles offers a fun way to validate your bioimage analysis expertise. 🎯🔬


If you solved all levels, but still can't get enough, just drag-and-drop a microscopy image on the playground and experiment with. 

## Acknowledgments

We are happy that we could use example images from others in this repository.
* Maize from [David Leglands' Mathematical Morphology Training Materials](https://github.com/dlegland/mathematical_morphology_with_MorphoLibJ/) licensed [CC-BY 4.0](https://github.com/dlegland/mathematical_morphology_with_MorphoLibJ/blob/master/LICENSE).
* A cropped version of the [human mitosis dataset in scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.human_mitosis).
* A single slice of the [cells3d dataset in scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.cells3d).
* The blobs image from [ImageJ](https://imagej.net).

Under the hood we are using
* [pyclesperanto](https://github.com/clEsperanto/pyclesperanto) and [scikit-image](https://scikit-image.org/) for the image processing and
* [pygame](https://www.pygame.org/news) for the user interface.

## Contributing

You can also save image processing workflows using the `Save` button and submit them as pull-request to create a new level! Just reach out by creating a [github-issue](https://github.com/haesleinhuepf/bia-bubbles/issues).

