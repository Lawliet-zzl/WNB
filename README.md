# Code for "Weighted Non-IID Batch: An Approach from Data Perspective for Out-of-distribution Detection"

## requirement
* conda install pytorch=1.1
* conda install gpustat
* conda install matplotlib
* conda install scikit-learn
* conda install scipy
* conda install pandas
* conda install requests
* conda install torchvision=0.3
* conda install tqdm
* conda install IPython
* conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)

Our codes will download the two in-distribution datasets automatically.

### Out-of-Distribtion Datasets
* [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [StanfordDogs120](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* [OxfordPets37](https://www.robots.ox.ac.uk/~vgg/data/pets/)
* [DTD47](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

Each out-of-distribution dataset should be put in the corresponding subdir in ./data/ + 'OOD name'

## Train and Test
Run the script [demo.sh](demo.sh). 
