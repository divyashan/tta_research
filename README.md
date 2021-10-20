# Test-Time Augmentation Research
Starter code for test-time augmentation related research. 

1. Clone this repo

```
git clone https://github.com/divyashan/tta_research.git
```

You may need to set up an SSH key. You can generate an SSH key following the instructions [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) and add it to your account by following the instructions [here](https://docs.github.com/en/enterprise-server@3.0/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account). You'll have to do this for each server/location you SSH from.

2. Create a conda environment with libraries in ``requirements.txt``

```
conda create --name my-amazing-tta-project python=3.6
source activate my-amazing-tta-project
conda config --add channels pytorch
conda install --file conda_requirements.txt
```

You'll have to install the utee package by running the following command in the pytorch-playground directory:

```
python3 setup.py develop --user
```

3. Start a jupyter notebook by running this command at the root directory of the project:

```
jupyter notebook --no-browser
```

4. Add the following shortcut for SSH tunnels your ~/.bashrc file. You only need to do this once.

```
tunnel() {
    ssh -N -f -L localhost:9000:localhost:$2 <CSAIL-KERB>@$1.csail.mit.edu
}
```

5. Run ```tunnel <oreo,mars,ahoy,twix> 8888``` in your local command line. 

6. Open the notebook at localhost:8888, and run through the cells in Sandbox.ipynb!



