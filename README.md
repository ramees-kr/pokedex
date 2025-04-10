# Pokémon Image Classifier with Streamlit Pokédex

## 🔎 Objective

This project aims to build and deploy a deep learning-based image classification application using PyTorch and Streamlit. The application identifies Pokémon from images and displays relevant Pokédex information interactively.

## ✨ Features

- **Image Classification:** Identifies Pokémon species from provided test images.
- **Model Selection:** Allows users to choose between different trained deep learning models for inference (SimpleCNN, ResNet18, DenseNet121, MobileNetV2).
- **Interactive Interface:** Built with Streamlit, featuring:
  - A left sidebar for configuration (model and image selection).
  - A main panel displaying the selected image and the prediction results (predicted Pokémon name and confidence score).
  - A right panel displaying Pokédex information (type, base stats, description) for the predicted Pokémon.
- **Pokédex Integration:** Fetches and displays data for the predicted Pokémon from a `pokedex.json` file.

## 📸 Demo / Screenshots

_(Placeholder: Add screenshots of your Streamlit application here)_

- _Screenshot of the main interface_
- _Screenshot showing model selection_
- _Screenshot showing Pokédex info_

## 📁 Project Structure

The project follows this directory structure:

```
pokemon_recognition/
├── app.py                 # Main Streamlit application script
├── class_names.txt        # List of Pokémon class names for the model
├── pokedex.json           # Pokémon metadata file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── models/                # Folder containing trained model weights (.pth files)
│   ├── best_model_SimpleCNN.pth
│   ├── best_model_ResNet18.pth
│   ├── best_model_DenseNet121.pth
│   └── best_model_MobileNetV2.pth
├── test_images/           # Folder containing sample images for testing
│   ├── Pikachu_abc123.jpg
│   └── ...
├── .venv/                 # Virtual environment folder (if used)
├── Streamlit_Application_Plan.txt # Original planning document
└── Training_Models_Plan.txt # Original training documentation
```

_(Note: Model file names in `models/` might vary slightly based on your saving convention)_

## 🧠 Models

Several Convolutional Neural Network (CNN) models were trained for classification:

- **SimpleCNN:** A basic custom CNN architecture.
- **ResNet18:** Pretrained ResNet18 with transfer learning.
- **DenseNet121:** Pretrained DenseNet121 with transfer learning (identified as the best performing model in initial tests).
- **MobileNetV2:** Pretrained MobileNetV2 with transfer learning.

**Training Details:**

- **Dataset:** Trained on a dataset of approximately 7,000 labeled Pokémon images, structured by class folders (e.g., `PokemonData/Pikachu/`). Dataset sourced from Kaggle: [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data).
- **Preprocessing:** Images resized to 128x128 pixels. Training included augmentations like random crops, flips, rotation, and color jitter. Inference uses resizing and tensor conversion.
- **Framework:** PyTorch.
- **Training:** Models were trained for 12 epochs.

## 📊 Data Sources

- **Pokémon Image Dataset:** Sourced from Kaggle: [Pokémon Classification Dataset](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/data) (Approx. 7,000 images across 150+ classes).
- **Pokédex Data:** `pokedex.json` obtained from a public source: [Purukitto/pokemon-data.json](https://github.com/Purukitto/pokemon-data.json/blob/master/pokedex.json).

## 🛠️ Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/ramees-kr/pokedex.git
    cd pokemon_recognition
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure you have the necessary data files:**
    - `class_names.txt` in the root directory.
    - `pokedex.json` in the root directory.
    - Trained model `.pth` files inside the `models/` directory.
    - Sample images inside the `test_images/` directory.

## 🚀 Running the Application

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```
2.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py --server.fileWatcherType none
    ```

3.  The application should open in your web browser automatically.

## 📋 Usage

1.  Use the **left sidebar** to select the classification model you want to use from the dropdown menu.
2.  Use the second dropdown in the left sidebar to choose a Pokémon image from the `test_images/` folder. A preview will appear below.
3.  The **main panel** will display the selected image and the model's prediction (Pokémon name and confidence score).
4.  The **right panel** will automatically display Pokédex information (Type, Stats, Description) for the predicted Pokémon, if available in the `pokedex.json` file.

## 💡 Future Enhancements

Potential improvements and future steps for this project include:

- Implementing drag-and-drop image upload for user-provided images.
- Adding real-time prediction capabilities using webcam input.
- Enhancing the `pokedex.json` file with more detailed descriptions or Pokémon sprites/images.
- Deploying the application to a platform like Streamlit Cloud or Hugging Face Spaces.
- Visualizing confidence scores for the top predicted classes (e.g., using a bar chart).

## 🙏 Acknowledgements

- Thanks to the creators of the Pokémon image dataset and the Pokédex JSON data.
- Built using PyTorch and Streamlit.
