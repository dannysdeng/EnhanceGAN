# Aesthetic-Driven Image Enhancement by Adversarial Learning
This is the demo code for the paper *Aesthetic-Driven Image Enhancement by Adversarial Learning* by *Deng et al.* at *ACMMM 2018*. 
([ArXiv](https://arxiv.org/abs/1707.05251)).

The code is based on: 
[1] https://github.com/soumith/dcgan.torch
[2] https://github.com/qassemoquab/stnbhwd

## Prerequisite
###### 1. Install torch7 (lua)
```
   git clone https://github.com/torch/distro.git ~/torch --recursive
   cd ~/torch; bash install-deps;
   ./install.sh;
   source ~/.bashrc;
```
###### 2. Install the standard STN package
```
git clone https://github.com/qassemoquab/stnbhwd.git; cd stnbhwd && luarocks make stnbhwd-scm-1.rockspec;
```
###### 3. Copy the customized STN files to the torch folder path.
```
   cp ./stn/*.lua ~/torch/install/share/lua/5.1/stn/
```
###### 4. Other lua dependencies
```
   luarocks install cudnn; # (The current version requires cudnn 5.0)
   luarocks install tds;
   luarocks install nngraph;
   luarocks install dpnn;
   luarocks install matio; # (Make sure that the shared libraries (libmatio.so or libmatio.dylib) are in your library path, e.g., sudo apt-get install libmatio2)
   luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```

###### 5. Display UI
Optionally, for displaying images during training and generation, we will use the display package.
Start the server with: th -ldisplay.start
Open this URL in your browser: http://localhost:8000

## Download pre-trained models and images:
You may download the models and our images (from [AVA dataset](http://vislab.berkeleyvision.org/datasets.html)) and extract to the repective directories from 
[Google_Drive](https://drive.google.com/open?id=13ay08vLY2OezNbDxj-nSMRGcMy2qW8z6)
or
[Baidu Pan](https://pan.baidu.com/s/1kMu4WdWqeyRScMYKTSUN5w)


## To run enhancement for an input image:
```
   cd ./ACMMM_2018_release;
   ./run_single_image.sh
   
   (For CPU mode, use "--gpu 0"  in ./run_single_image.sh. 
    Note that you might need to download "model_best_cpu.t7" and put it in the "./ACMMM_2018_release/checkpoints/" folder.)
```
## To run training (~11 GB of GPU memory required):
```
  cd ./ACMMM_2018_release;
  ./train_my_own.sh
  ```
