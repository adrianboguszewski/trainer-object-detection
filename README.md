# Trainer Package: Train Object Detection Model
This project demonstrates an object detection *trainer package* that works on object detection datasets such 
as "coco-2017" and "midwest-vehicle-detection". 

**NOTE: This readme will only contain the most basic steps to get the trainer package up and running.
For more details on trainer packages and Training-aaS checkout the
 [trainer-classification README](https://github.com/milestone-hafnia/trainer-classification?tab=readme-ov-file#trainer-package-train-image-classification-model).**

# Quick Start: No-Code Model Training
For a quick start to launch model training using this trainer package without any code changes, follow the steps below:

1. Go to the [experiments dashboard](https://hafnia.milestonesys.com/dashboard/training-aas/experiments) and press 
the "Create Experiment" button.
2. Select the dataset you want to use (e.g. "coco-2017" or "midwest-vehicle-detection").
3. Upload the pre-built trainer package `trainer.zip` located in the root folder of this repo. Use this [link](https://raw.githubusercontent.com/milestone-hafnia/trainer-object-detection/main/trainer.zip) to download. 
4. Provide the training command: `python scripts/train.py` 
5. Select your desired training configuration (e.g., "Free Tier" or "Professional").
6. Press "Create Experiment" to launch the training job.
7. Monitor the training job in the experiments dashboard.

# Trainer Package Development
If you want to extend or update this trainer package or if you want to develop your own trainer package follow below steps.

## Setup and Install Trainer Package Locally
First you will need to clone the repo and install dependencies in a virtual environment here using `uv` as package manager.
```bash
# Download uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
cd [SOME_DESIRED_PATH]
git clone https://github.com/milestone-hafnia/trainer-object-detection

# Install dependencies in virtual environment
cd trainer-object-detection
uv sync

source .venv/bin/activate
```

## Run and Debug Trainer Package in VS-code
The trainer package is created to work in a local environment with VS-code. 
1. Open the project folder in VS-code.
2. Add python Interpreter from the virtual environment `.venv/bin/python`.
   Press `Ctrl+Shift+P` and search for `Python: Select Interpreter`.
3. In the debug panel, select the configuration `Model Training` and press F5 or click the green play button 
   to start debugging. 


## Build Trainer Package Zip file
You can also create the trainer package zip file and launch experiments from the command line using the `hafnia` cli.

```bash
# Update `trainer.zip` from command line
hafnia trainer create-zip .
```

## Launch Experiment Directly from Command Line
Packing the trainer package as a zip file and adding it through the Hafnia Portal can be cumbersome. 

Instead, you package your trainer package and launch experiments directly from the command line using this command:
```bash
# Configure hafnia cli
hafnia configure

# Launch experiment from command line
hafnia experiment create --dataset midwest-vehicle-detection --trainer-path . --cmd "python scripts/train.py --epochs 1"
```

## Build and Launch Trainer Package Locally
To test the trainer package locally using docker, you can follow the steps below:

```bash
    # Create 'trainer.zip' from source folder
    hafnia trainer create-zip .
    
    # Build the docker image locally from a 'trainer.zip' file
    hafnia runc build-local trainer.zip

    # Execute the docker image locally with a desired dataset
    hafnia runc launch-local --dataset midwest-vehicle-detection  "python scripts/train.py"
```