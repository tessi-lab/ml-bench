# ML-Benchmark

### Description

This projet aims to compare performance
of computers with the same type of training.

It's based on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
There are some parameters that can be tuned.

**Epoch**: number, positive integer. By default 20.

**Neural network size**: controls the number of filters applied on 2 first layers
 - basic : [4, 8] enough to make the network converge,
 - normal: [8, 16] enough to make the network converge with more data,
 - heavy: [16, 32] too many filters for a such a simple task,
 - too-heavy: [64, 128] idem, but quadruple, will probably not converge,
 - stupid: [128, 256] idem, but stupidly huge,
 - insane: [256, 512] idem, but insanely huge.

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

`python -m venv ml-bench`

Use the right activate for your shell (for example bash)

`source ml-bench/bin/activate`

Download sources from Github:

`git clone https://github.com/tessi-lab/ml-bench.git`

`cd ml-bench`

Install the packages for your system.
For macOS with Apple Silicon see this [article](https://developer.apple.com/metal/tensorflow-plugin/).
Example: 

`python -m pip install -r requirements.txt`

Launch the tool : 

`python benchmark.py`
