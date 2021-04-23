COMP532 - CA2
=============
By Thomas Rex Greenway, Kit Bower-Morris, Nicholas Bryan

Implementation of a Deep Q-Learning model on an OpenAI Gym environment interacting with a TensorFlow 
neural newtwork.

How To Run
----------
(This program was run in VSCode with the dependencies specified below)

- Please run the single source code python file (CartPole_DQN.py) as the main file to reproduce 
    results.
- Hyper-Parameters can be adjusted within main(). 
- The program first performs a given number of training episodes over the model, plotting the moving
    average of rewards during this phase. Upon closing of the produced graph the program then proceeds
    to run 50 rendered episodes on the trained network.

Dependencies
------------
Python 3.8.4rc1
TensorFlow 2.4.1
OpenAI gym 0.18.0
matplotlib 3.4.1
numpy 1.19.5

Notes
-----
Please be aware an error may be produced upon the completion and closure of the environment render
window depending on the users operating system. This has no impact upon the running of the program.