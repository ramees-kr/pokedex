import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, densenet121, mobilenet_v2
from torchvision.models import ResNet18_Weights, DenseNet121_Weights, MobileNet_V2_Weights
from PIL import Image
import os
import json

# --------------------------------------------------
# Configuration and Setup
# --------------------------------------------------
st.set_page_config(page_title="Pokémon Classifier Demo", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder and file paths
MODELS_DIR = "./models"                # Your folder with .pth model files
TEST_IMAGES_DIR = "./test_images"      # Folder with test images for Browse
CLASS_NAMES_FILE = "class_names.txt"   # File with one Pokémon name per line
POKEDEX_FILE = "pokedex.json"          # Pokedex file obtained from the internet

# Image parameters
IMG_SIZE = 128  # Must match your training/resizing settings

# Model options: we expect saved model weight files like best_model_<model_name>.pth
all_model_options = ["SimpleCNN", "ResNet18", "DenseNet121", "MobileNetV2"]
available_models = []
for model_name in all_model_options:
    # Adjust path logic if model names differ slightly (e.g., no 'best_model_' prefix)
    model_path_options = [
        os.path.join(MODELS_DIR, f"best_model_{model_name}.pth"),
        os.path.join(MODELS_DIR, f"model_{model_name}.pth"), # Add variation if needed
        os.path.join(MODELS_DIR, f"{model_name}.pth")        # Add variation if needed
    ]
    for model_path in model_path_options:
        if os.path.exists(model_path):
            available_models.append(model_name)
            break # Found one, move to next model name

if not available_models:
    st.error("No model weight files found in the 'models' folder matching expected patterns!")
    st.stop()

# --------------------------------------------------
# Model Architecture Definitions (must match training)
# --------------------------------------------------
# Ensure these exactly match the classes used during training saved in .pth files
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (IMG_SIZE // 4) * (IMG_SIZE // 4), 128), # Corrected calculation based on pooling
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class PretrainedDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedDenseNet, self).__init__()
        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        # Freeze parameters in the feature extractor
        for param in self.model.features.parameters():
             param.requires_grad = False
        # Replace the classifier layer
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

class PretrainedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedMobileNetV2, self).__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
         # Freeze parameters
        for param in self.model.parameters():
             param.requires_grad = False
        # Replace the classifier layer
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)
    def forward(self, x):
        return self.model(x)

# Map model names to their class definitions
model_dict = {
    "SimpleCNN": SimpleCNN,
    "ResNet18": PretrainedResNet18,
    "DenseNet121": PretrainedDenseNet,
    "MobileNetV2": PretrainedMobileNetV2,
}

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

# Load class names from class_names.txt
@st.cache_data
def load_class_names(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: # Ensure UTF-8 encoding
            lines = f.readlines()
        # Strip whitespace and filter out empty lines
        names = [line.strip() for line in lines if line.strip()]
        return names
    except Exception as e:
        st.error(f"Error loading class names from '{file_path}': {e}")
        return []

class_names = load_class_names(CLASS_NAMES_FILE)
if not class_names:
    st.error("Failed to load class names or file is empty.")
    st.stop()
num_classes = len(class_names)


# Load and convert pokedex.json into a dictionary keyed by english name
@st.cache_data
def load_pokedex(file_path):
    try:
        # Corrected line with UTF-8 encoding specified
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        # Convert list to dict: keys are the English names, case-insensitive for robustness
        pokedex = {entry["name"]["english"].lower(): entry for entry in data if isinstance(entry, dict) and "name" in entry and "english" in entry["name"]}
        return pokedex
    except FileNotFoundError:
        st.error(f"Error: Pokedex file not found at '{file_path}'")
        return {}
    except json.JSONDecodeError:
         st.error(f"Error: Pokedex file '{file_path}' is not valid JSON.")
         return {}
    except Exception as e:
        st.error(f"Error loading pokedex data from '{file_path}': {e}")
        return {}

pokedex_data = load_pokedex(POKEDEX_FILE)
if not pokedex_data:
     st.warning(f"Could not load Pokedex data from {POKEDEX_FILE}. Pokedex info will be unavailable.")


# Function to find the correct model file path based on common naming conventions
def find_model_path(model_name):
    potential_paths = [
        os.path.join(MODELS_DIR, f"best_model_{model_name}.pth"),
        os.path.join(MODELS_DIR, f"model_{model_name}.pth"),
        os.path.join(MODELS_DIR, f"{model_name}.pth")
    ]
    for path in potential_paths:
        if os.path.exists(path):
            return path
    return None # Return None if no matching file is found

# Function to load a selected model
@st.cache_resource(show_spinner="Loading Model...")
def load_model(model_name):
    model_class = model_dict.get(model_name)
    if model_class is None:
        st.error(f"Model architecture '{model_name}' is not defined in the script.")
        return None

    model_path = find_model_path(model_name)
    if model_path is None:
        st.error(f"Could not find a weights file for model '{model_name}' in '{MODELS_DIR}'. Looked for patterns like 'best_model_{model_name}.pth', 'model_{model_name}.pth', '{model_name}.pth'.")
        return None

    try:
        model = model_class(num_classes=num_classes)
        # Load the state dict using map_location for CPU fallback
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        st.success(f"Loaded model: {model_name}")
        return model
    except FileNotFoundError:
         st.error(f"Error: Model weights file not found at '{model_path}'")
         return None
    except Exception as e:
        st.error(f"Error loading model weights from '{model_path}': {e}")
        return None


# Function to preprocess input image for inference
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize to expected input size
        transforms.ToTensor()                    # Convert PIL Image to tensor
    ])
    # Add batch dimension (B, C, H, W)
    return transform(image).unsqueeze(0)

# --------------------------------------------------
# Sidebar: Configuration
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    # Check if available_models is populated before creating selectbox
    if available_models:
        selected_model_name = st.selectbox("Select Evaluation Model", available_models)
    else:
        selected_model_name = None # Handle case where no models were loaded

    st.header("🖼️ Test Image Explorer")
    # Ensure TEST_IMAGES_DIR exists
    if os.path.isdir(TEST_IMAGES_DIR):
        test_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_files:
            selected_test_file = st.selectbox("Choose a test image", test_files)
            test_image_path = os.path.join(TEST_IMAGES_DIR, selected_test_file)
            # Corrected st.image call for sidebar
            st.image(test_image_path, caption=selected_test_file, use_container_width=True)
        else:
            st.warning(f"No test images found in '{TEST_IMAGES_DIR}'.")
            selected_test_file = None
    else:
        st.error(f"Test images directory not found: '{TEST_IMAGES_DIR}'")
        selected_test_file = None

# --------------------------------------------------
# Main Area Layout (2 columns: Main Content | Pokedex Info)
# --------------------------------------------------
st.title("⚡ Pokémon Classifier Demo ⚡")

# Create two columns: 2/3 for main content, 1/3 for Pokedex
col1, col2 = st.columns([2, 1])

predicted_name = None # Initialize predicted_name

with col1: # Main Content Area
    st.header("Image & Prediction")
    if selected_test_file and selected_model_name:
        image_path = os.path.join(TEST_IMAGES_DIR, selected_test_file)
        try:
            image = Image.open(image_path).convert("RGB")
            # Corrected st.image call for main area - use 'width' instead of deprecated param
            st.image(image, caption="Selected Image", width=400) # Adjust width as needed

            input_tensor = preprocess_image(image)
            model = load_model(selected_model_name) # Load the selected model

            if model is not None:
                with st.spinner("Classifying..."):
                    with torch.no_grad():
                        output = model(input_tensor.to(DEVICE))
                        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        pred_idx = int(probabilities.argmax())
                        confidence = probabilities[pred_idx] * 100

                if 0 <= pred_idx < len(class_names):
                    predicted_name = class_names[pred_idx]
                    st.subheader("Prediction Result")
                    st.success(f"**Predicted Pokémon:** {predicted_name}")
                    st.info(f"**Confidence:** {confidence:.2f}%")
                    st.progress(int(confidence))
                else:
                     st.error(f"Prediction index {pred_idx} is out of bounds for class names list (length {len(class_names)}).")
            # Error handled within load_model

        except FileNotFoundError:
            st.error(f"Selected image not found: {image_path}")
        except Exception as e:
             st.error(f"An error occurred processing the image or predicting: {e}")

    elif not selected_model_name:
         st.warning("Please select a model from the sidebar.")
    else:
        st.info("Please select a test image from the sidebar.")


with col2: # "Right Sidebar" for Pokedex Info
    st.header("Pokédex Information")
    if predicted_name and pokedex_data:
        # Fetch details using lowercase for robustness
        info = pokedex_data.get(predicted_name.lower())
        if info:
            st.markdown(f"### {info['name']['english']}")

            # Display image if available in Pokedex data
            if "image" in info and "thumbnail" in info["image"]:
                 st.image(info["image"]["thumbnail"], caption=f"{info['name']['english']} Thumbnail")

            if "species" in info:
                st.write(f"**Species:** {info['species']}")
            if "type" in info:
                st.write("**Type:** " + ", ".join(info["type"]))

            # Display base stats in a more structured way
            if "base" in info:
                st.write("**Base Stats:**")
                stats_data = {"Stat": list(info["base"].keys()), "Value": list(info["base"].values())}
                st.dataframe(stats_data, use_container_width=True)

            if "description" in info:
                st.write("**Description:**")
                st.write(info["description"])
        else:
            st.info(f"No Pokédex information found for '{predicted_name}'.")
    elif not pokedex_data:
         st.info("Pokedex data could not be loaded. Information unavailable.")
    else:
        st.info("Select an image and model to see Pokédex information.")

# Footer or additional info
st.markdown("---")
st.markdown("App demonstrating Pokémon image classification.")