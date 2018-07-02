1. Install torch7 (lua)
   git clone https://github.com/torch/distro.git ~/torch --recursive
   cd ~/torch; bash install-deps;
   ./install.sh;
   source ~/.bashrc;

2. git clone https://github.com/qassemoquab/stnbhwd.git; cd stnbhwd && luarocks make stnbhwd-scm-1.rockspec;
3. Copy the customized STN to the torch folder path.
   cp ./stn/*.lua ~/torch/install/share/lua/5.1/stn/

4. luarocks install cudnn; # (The current version requires cudnn 5.0)
   luarocks install tds;
   luarocks intsall nngraph;
   luarocks install dpnn;
   luarocks install matio; # (Make sure that the shared libraries (libmatio.so or libmatio.dylib) are in your library path, e.g., sudo apt-get install libmatio2)
   luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec


5. Display UI
Optionally, for displaying images during training and generation, we will use the display package.
Start the server with: th -ldisplay.start
Open this URL in your browser: http://localhost:8000

6. To run a input image:
   cd ./ACMMM_2018_release;
   ./run_single_image.sh

7. To run training (~11 GB of GPU memory required):
  cd ./ACMMM_2018_release;
  ./train_my_own.sh
