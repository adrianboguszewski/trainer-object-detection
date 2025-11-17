# Trainer Package: Train Object Detection Model
This project demonstrates an object detection trainer package for Hafnia Training-as-a-Service (Training-aaS), compatible with object detection datasets such as "coco-2017" and "midwest-vehicle-detection". 


> **Note:** This README covers the essential steps to get started. For more details on trainer packages and Training-aaS, visit the [trainer-classification README](https://github.com/milestone-hafnia/trainer-classification?tab=readme-ov-file#trainer-package-train-image-classification-model).

## Quick Start: No-Code Model Training

Launch model training instantly without code modifications using the Hafnia Training-aaS platform:

### Steps:

1. **Access the Dashboard**  
   Navigate to the [experiments dashboard](https://hafnia.milestonesys.com/dashboard/training-aas/experiments) and click "Create Experiment"

2. **Select Dataset**  
   Choose your target dataset (e.g., `coco-2017` or `midwest-vehicle-detection`)

3. **Upload Trainer Package**  
   Download and upload the pre-built `trainer.zip` from: [trainer.zip](https://raw.githubusercontent.com/milestone-hafnia/trainer-object-detection/main/trainer.zip)

4. **Configure Training**  
   - **Training command:** `python scripts/train.py`
   - **Configuration:** Select "Free Tier" or "Professional" based on your needs

5. **Launch & Monitor**  
   Click "Create Experiment" and monitor progress in the dashboard

---

# Trainer Package Development
If you want to extend or update this trainer package, or develop your own trainer package, follow the steps below.

## Setup and Install Trainer Package Locally
First, you need to clone the repository and install dependencies in a virtual environment using `uv` as the package manager.
```bash
# Download uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
cd [SOME_DESIRED_PATH]
git clone https://github.com/milestone-hafnia/trainer-object-detection

# Install dependencies in virtual environment
cd trainer-object-detection
uv sync

source .venv/bin/activate
```

## Run and Debug Trainer Package in VS Code
The trainer package is designed to work in a local environment with VS Code. 
1. Open the project folder in VS Code.
2. Add the Python interpreter from the virtual environment `.venv/bin/python`.
   Press `Ctrl+Shift+P` and search for `Python: Select Interpreter`.
3. In the debug panel, select the configuration `Model Training` and press F5 or click the green play button 
   to start debugging. 


## Build Trainer Package Zip File
Create a trainer package zip file using the Hafnia CLI.

```bash
# Update `trainer.zip` from command line
hafnia trainer create-zip .
```

## Launch Experiment Directly from Command Line
Packing the trainer package as a zip file and uploading it through the Hafnia Web-portal can be become cumbersome, when
running multiple experiments or making frequent updates to the trainer package.

Instead, you can package your trainer package and launch experiments directly from the command line using this command:
```bash
# Configure hafnia CLI (Only done once)
hafnia configure

# Launch experiment from command line
hafnia experiment create --dataset midwest-vehicle-detection --trainer-path . --cmd "python scripts/train.py --epochs 1"
```
In above example the `--trainer-path` argument points to the local trainer package folder, the `midwest-vehicle-detection` dataset will be used, and the training commands specifies to only run for 1 epoch.

## Build and Launch Trainer Package Locally
To test the trainer package locally in Docker, follow the steps below:

```bash
# Create 'trainer.zip' from source folder
hafnia trainer create-zip .

# Build the Docker image locally from a 'trainer.zip' file
hafnia runc build-local trainer.zip

# Execute the Docker image locally with a desired dataset
hafnia runc launch-local --dataset midwest-vehicle-detection  "python scripts/train.py"
```