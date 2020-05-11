# Navigation project

![banana](images/banana-init.gif)


## Project Details

The goal here is to train an agent to collect as many yellow bananas as possible in a
limited amount of steps while avoiding the blue ones along the way. The task, in
order to solve the environment, is to achieve an average score of `+13` over 100 consecutive episodes.

The following are the details of the environment used in this project.

```
Unity brain name: BananaBrain
        Number of Visual Observations (per agent): 0
        Vector Observation space type: continuous
        Vector Observation space size (per agent): 37
        Number of stacked Vector Observation: 1
        Vector Action space type: discrete
        Vector Action space size (per agent): 4
        Vector Action descriptions: , , ,
```

As we can see above the state space is continuous and has 37 dimensions. The
action space is discrete and has 4 actions available.


## Getting Started

1. Visit the following link for instructions on how to set up your environment: [instructions](https://github.com/udacity/deep-reinforcement-learning#dependencies)\
2. Run `pip install -r requirements.txt`, using the virtual environment created in the previous step, to install the remaining of the required dependencies.

3. The environment included in this repository is for the Linux platform only. See
below for environments for other platforms:
  * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


## Instructions

- Once the requirements detailed above are satisfied run the following command to
train the Agent:\
`python banana_finder.py`
- To run the Agent in eval mode we need to pass a checkpoint file as a CLI
argument.
- Run the following command for further details:\
`python banana_finder.py -h`
