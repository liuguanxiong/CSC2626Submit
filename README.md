# CSC2626Submit

A pytorch implementation to train the conditional imitation learning policy in "End-to-end Driving via Conditional Imitation Learning" and "CARLA: An Open Urban Driving Simulator".

## Requirements
python 3.6    
pytorch > 0.4.0    
tensorboardX    
numpy    
torchvision    
cuda    
h5py    
PIL

## Train
**train-dir** and **eval-dir** should point to where the [Carla dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) located.
Please put the train dataset and eval dataset in two subfolders.
```
$ python main.py --batch-size
150
--workers
4
--train-dir
"data/SeqTrain/"
--eval-dir
"data/SeqVal/"
--id
training
--gpu
0
```
## evaluate with Carla Simulator
```
Have the Carla simulator running on server mode. You can run with $CarlaUE4.exe -carla-server -windowed -benchmark-fps=20
$python run_cil.py
