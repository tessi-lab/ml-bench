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

_With a M1 Apple Silicon_ :

    bash Miniforge3-MacOSX-arm64.sh
    . miniforge3/bin/activate
    conda install -c apple tensorflow-deps
    pip install tensorflow-macos
    pip install tensorflow-metal
    conda install -c apple wxpython
    conda install -c apple opencv
    pip install python-mnist

_With an Intel Mac_:

Example: 

`python -m pip install -r requirements_macos_intel.txt`

Launch the tool : 

`python benchmark.py`

or

`pythonw benchmark.py`

### Results for size normal and 20 epochs
#### AlienWare Ryzen Edition AMD Ryzen 9 5950X 16-Core 64 GB RAM, SSD PM9A1 2 TB
- LOAD_TRAIN: 3380 ms
- CREATE_TRAIN: 6354 ms
- LOAD_TEST: 547 ms
- CREATE_TEST: 569 ms
- TRAIN: 116817 ms

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

#### iMac 2020 24' M1 - 16GB RAM - SSD 1 TB Metal (8 GPU Cores)
- LOAD_TRAIN: 3326 ms
- CREATE_TRAIN: 6384 ms
- LOAD_TEST: 523 ms
- CREATE_TEST: 540 ms
- TRAIN: 129372 ms

#### iMac 2020 24' M1 - 16GB RAM - SSD 1 TB CPU Only
(same computer uninstall tensorflow-metal package)

- LOAD_TRAIN: 3242 ms
- CREATE_TRAIN: 6304 ms
- LOAD_TEST: 518 ms
- CREATE_TEST: 535 ms
- TRAIN: 217383 ms

#### MacBook 2019 16' - i9 - 32 GB RAM - SSD 1 TB on CPU and battery (turbo boost switcher)
- LOAD_TRAIN: 8470 ms
- CREATE_TRAIN: 16285 ms
- LOAD_TEST: 1308 ms
- CREATE_TEST: 1373 ms
- TRAIN: 1105780 ms 

#### MacBook 2019 16' - i9 - 32 GB RAM - SSD 1 TB on CPU and plugged to the wall
- LOAD_TRAIN: 4760 ms
- CREATE_TRAIN: 9552 ms
- LOAD_TEST: 746 ms
- CREATE_TEST: 782 ms
- TRAIN: 437925 ms

#### MacBook 2019 16' - i9 - 32 GB RAM - SSD 1 TB on GPU 5500M (8GB) and battery (turbo boost switcher)
- LOAD_TRAIN: 8492 ms
- CREATE_TRAIN: 17309 ms
- LOAD_TEST: 1343 ms
- CREATE_TEST: 1409 ms
- TRAIN: 222868 ms

#### MacBook 2019 16' - i9 - 32 GB RAM - SSD 1 TB on GPU 5500M (8GB) and plugged to the wall
- LOAD_TRAIN: 4625 ms
- CREATE_TRAIN: 10190 ms
- LOAD_TEST: 722 ms
- CREATE_TEST: 759 ms
- TRAIN: 153928 ms

#### MacBook 2021 16' - M1 Max - 64 GB RAM - SSD 2 TB and plugged to the wall
- LOAD_TRAIN: 3224 ms
- CREATE_TRAIN: 6292 ms
- LOAD_TEST: 517 ms
- CREATE_TEST: 533 ms
- TRAIN: 111895 ms
