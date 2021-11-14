# ML-Benchmark

### Visuals
Main Screen

![Main screen](/assets/screen-capture1.jpg?raw=true "Main screen")


Test Screen

![Test screen](/assets/screen-capture2.jpg?raw=true "Test screen")

### Description

This projet aims to compare performance
of computers with the same type of training.

It's based on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
There are some parameters that can be tuned.

**Epoch**: number, positive integer. By default 20.

**Neural network size**: controls the number of filters applied on 2 first layers
 - `light : `[2, 4],
 - `basic : `[4, 8] enough to make the network converge,
 - `normal: `[8, 16] enough to make the network converge with more data,
 - `heavy : `[16, 32] too many filters for a such a simple task,
 - `too-heavy:`[64, 128] idem, but quadruple, will probably not converge,
 - `stupid: `[128, 256] idem, but stupidly huge,
 - `insane: `[256, 512] idem, but insanely huge.

The button **Launch training !** does just that, each step is timed.
It will display something like  :
 - 'load_test'  745.21 ms
 - 'train'  13899.57 ms

Once training is over you can copy paste the result in a file.
Then you may click **Play with it !** to draw digits in the black area
and see the result in the title bar.
Right click removes the last line drawn.

### Installation
Install python 3.8 for your system, create a virtual-env:

`python3 -m venv venv-ml-bench`

Use the right activate for your shell (for example bash)

`source venv-ml-bench/bin/activate`

Download sources from Github:

`git clone https://github.com/tessi-lab/ml-bench.git`

`cd ml-bench`

Install the packages for your system.
For macOS with Apple Silicon see this [article](https://developer.apple.com/metal/tensorflow-plugin/).

With a M1 Apple Silicon :
`bash Miniforge3-MacOSX-arm64.sh`
`. miniforge3/bin/activate`
`conda install -c apple tensorflow-deps`
`pip install tensorflow-macos`
`pip install tensorflow-metal`
`conda install -c apple wxpython`
`conda install -c apple opencv`
`pip install python-mnist`

With an Intel Mac:

Example: 

`python -m pip install -r requirements.txt`

Launch the tool : 

`python benchmark.py`

### Results for size normal and 20 epochs
#### iMac 2015, core i5, AMD Radeon R9 M390 2 Go, 24 GB RAM, SSD 1TB
##### Run on CPU only (requirements.txt)
- 'load_train'  5802.60 ms
- 'create_train'  12206.21 ms
- 'load_test'  943.89 ms
- 'create_test'  991.74 ms
- 'train'  328884.54 ms

##### Run on GPU + CPU (requirements_macos_intel.txt)
- 'load_train'  5908.59 ms
- 'create_train'  11447.35 ms
- 'load_test'  944.63 ms
- 'create_test'  992.16 ms
- 'train'  263666.66 ms

#### iMac 2020 24' M1 - 16GB RAM - SSD 1 TB
- LOAD_TRAIN: 3359 ms
- CREATE_TRAIN: 6436 ms
- LOAD_TEST: 541 ms
- CREATE_TEST: 558 ms
- TRAIN: 138559 ms