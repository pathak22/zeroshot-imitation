## Zero-Shot Visual Imitation ##
#### In ICLR 2018 [[Project Website]](https://pathak22.github.io/zeroshot-imitation/) [[Videos]](http://pathak22.github.io/zeroshot-imitation/index.html#demoVideos)

Deepak Pathak\*, Parsa Mahmoudieh\*, Guanghao Luo\*, Pulkit Agrawal\*, Dian Chen, <br/>Yide Shentu, Evan Shelhamer, Jitendra Malik, Alexei A. Efros, Trevor Darrell<br/>
University of California, Berkeley<br/>

<img src="https://pathak22.github.io/zeroshot-imitation/resources/turtle.gif" width="300">    <img src="https://pathak22.github.io/zeroshot-imitation/resources/baxter.gif" width="300">

This is the implementation for the [ICLR 2018 paper Zero Shot Visual Imitation](https://pathak22.github.io/zeroshot-imitation). We propose an alternative paradigm wherein an agent first explores the world without any expert supervision and then distills its experience into a goal-conditioned skill policy with a novel forward consistency loss. The key insight is the intuition that, for most tasks, reaching the goal is more important than how it is reached.

    @inproceedings{pathakICLR18zeroshot,
        Author = {Pathak, Deepak and
        Mahmoudieh, Parsa and Luo, Guanghao and
        Agrawal, Pulkit and Chen, Dian and
        Shentu, Yide and Shelhamer, Evan and
        Malik, Jitendra and Efros, Alexei A. and
        Darrell, Trevor},
        Title = {Zero-Shot Visual Imitation},
        Booktitle = {ICLR},
        Year = {2018}
    }

### 1) Installation and Usage

#### Requirements

```Shell
git clone -b master --single-branch https://github.com/pathak22/zeroshot-imitation.git
cd zeroshot-imitation/

# (1) Install requirements:
sudo apt-get install python-tk
virtualenv venv
source $PWD/venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install -r src/requirements.txt

# (2) Install Caffe: http://caffe.berkeleyvision.org/install_apt.html
git clone https://github.com/BVLC/caffe.git
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install libatlas-base-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install --no-install-recommends libboost-all-dev
cd caffe/  # edit Makefile.config
make all -j
make pycaffe
make test -j
make runtest -j
# Note: If you are using conda, then its easy:
# $ conda install -c conda-forge caffe
# $ conda install -c conda-forge opencv=3.2.0
```

#### Data setup
Data can be downloaded at [google drive link](https://drive.google.com/file/d/1pnX8gGqs5EQVjGy4Z6FZ5oZ9noDHlC_8/view). This is the same data as used in [Combining Self-Supervised Learning and Imitation for Vision-Based Rope Manipulation](https://ropemanipulation.github.io).

You will need the rope dataset from this download.

Then, download the AlexNet weights, bvlc_alexnet.npy from [here](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

- put rope data in data/datasets/rope9
- it is important to name it rope9!
- put bvlc_alexnet.npy in nets/bvlc_alexnet.npy

#### Training

```Shell
python -i train.py

# fwd_consist=True to turn foward consistency loss on,
# or leave it False for to just learn the inverse model
r = RopeImitator('name', fwd_consist=True)

# to train baseline, turn baseline_reg=True. note that fwd_consist should be turned on as well (historical accident)
r = RopeImitator('name', fwd_consist=True, baseline_reg=True)

# Restore old models, if any. default of model_name is just current model name
r.restore(iteration, model_name='name of old model')

# training
r.train(num_iters)

```

Note that the accuracies presented is not a good measure of real world performance. The purpose of forward consistency is to learn actions consistent with state transistions, which don't necessarily have to be the ground truth actions.

### 2) Other resources
  - [Paper](https://pathak22.github.io/zeroshot-imitation/resources/iclr18.pdf)
  - [Project Website](https://pathak22.github.io/zeroshot-imitation/)
  - [Videos](http://pathak22.github.io/zeroshot-imitation/index.html#demoVideos)
