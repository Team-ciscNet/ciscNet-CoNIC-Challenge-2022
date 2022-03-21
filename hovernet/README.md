# Training Instructions

Clone HoverNet repository from [GitHub](https://github.com/vqdang/hover_net/tree/conic) and switch to the 'conic' branch.
```
git clone https://github.com/vqdang/hover_net/tree/conic
cd hover_net
git switch -c conic origin/conic
```
Set up the virtual environment:
```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio===0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```
Create a folder for the results and download the pre-trained weights:
```
mkdir exp_output
mkdir exp_output/local
mkdir exp_output/local/data
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
mv resnet50-0676ba61.pth exp/output_local/resnet50.pth
```
Copy the files images.npy, gts.npy, counts.csv, ids_train_80.csv, ids_eval_80.csv and ids_val_80.csv into exp_output/local/data. Use the files from the folder "original_scale" and not the original files! Rename gts.npy to labels.npy.

Replace 'param/template.yaml' at HoverNet GitHub repository and add 'my_generate_split.py' to the HoverNet repo and train model as described but with our split generation method:
```
python my_generate_split.py
python run --gpu "0,1"
```
Copy the "baseline" directory. Otherwise it will be overwritten in the next training.

# Evaluation Instructions
Clone the CoNIC challenge repository 
```
git clone https://github.com/TissueImageAnalytics/CoNIC.git
```
and replace the jupyter notebooks within the notebooks directory.

Create a new virtual environment (installing into the hovernet ve does not work!)
```
conda env create --name hovernet_eval
conda activate hovernet_eval
conda install pip
pip install tiatoolbox
pip install jupyter ipywidgets
pip install openslide-python
sudo apt install libopenslide0
```
Run the 'plot_scores.ipynb' to find out the best performing HoverNet you trained and select the corresponding 'net_*.tar' file in the 'baseline.ipynb'.

Go to our conic-challenge-2022 repo and activate the corresponding virtual environment and run eval_hovernet.py.