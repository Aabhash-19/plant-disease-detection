# Plant Disease Detection

This repository provides a comprehensive solution for plant disease detection using deep learning models. The project is divided into two main parts:

* **Train\_plant\_disease.ipynb:** For training a deep learning model on plant disease images.
* **Test\_plant\_disease.ipynb:** For evaluating the trained model on test data.

## Project Structure

```
📂 Plant_Disease_Detection
├── 📄 Train_plant_disease.ipynb
├── 📄 Test_plant_disease.ipynb
└── 📄 README.md
```

## Requirements

* Python 3.x
* Jupyter Notebook
* TensorFlow/Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/Plant_Disease_Detection.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

* Open `Train_plant_disease.ipynb` in Jupyter Notebook.
* Run the cells step by step to train the model on your dataset.

### 2. Testing the Model

* Open `Test_plant_disease.ipynb` in Jupyter Notebook.
* Run the cells to evaluate the model on test images.

## Dataset

This dataset is taken from kaggle which is described to be recreated using offline augmentation from the original dataset. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

## Model

* The model is built using a convolutional neural network (CNN).
* Customizable hyperparameters like learning rate, batch size, and number of epochs.

## Results

* The trained model achieves high accuracy in detecting various plant diseases.
* Evaluation metrics are visualized using plots.

## License

This project is licensed under the MIT License.

