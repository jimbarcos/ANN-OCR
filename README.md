# 5x7 OCR Character Recognition System

A comprehensive neural network-based character recognition system with an intuitive GUI for training and testing 5x7 matrix character patterns.

## Key Features

### Neural Network Training
- **Comprehensive Training Data**: Pre-loaded with 87+ characters (0-9, A-Z, a-z, special characters)
- **Configurable Training Parameters**: Full control over epochs, batch size, learning rate, validation split
- **Customizable Architecture**: Adjustable number of neurons and dropout rates for each layer
- **Parameter Persistence**: Save and load training configurations as JSON files
- **Detailed Training Logs**: Real-time monitoring with epoch-by-epoch progress
- **Performance Analysis**: Automatic evaluation and recommendations

### Comprehensive Visualization
- **Training Progress**: Live plots of accuracy and loss during training
- **Network Processing**: See how the ANN processes your input through each layer
- **Activation Maps**: Visualize neuron activations in hidden layers
- **Output Probabilities**: Bar chart showing prediction confidence for each character with ASCII codes

### Interactive Drawing
- **5x7 Grid Canvas**: Click cells to toggle between black (1) and white (0)
- **Real-time Recognition**: Instant character prediction with confidence scores
- **Clear Function**: Reset canvas for new character drawing

### Character Set Selection
- **Flexible Training**: Choose specific character sets (Numbers, Uppercase, Lowercase, Special)
- **Dynamic Filtering**: Train and recognize only selected character types
- **Real-time Updates**: Character count updates as selections change
- **Parameter Persistence**: Character selections saved with training configurations

### Recognition Results
- **Character Prediction**: Displays the recognized character
- **ASCII Code**: Highlighted ASCII value in blue and bold for emphasis
- **Confidence Score**: Prediction confidence percentage

### Data Import
- **Excel File Support**: Import training data from Excel files
- **Flexible Format**: Supports various Excel formats (.xlsx, .xls)

### System Logging
- **Real-time Logs**: Monitor all system activities
- **Error Tracking**: View detailed error messages
- **Training Updates**: Track training progress and results

## Installation

1. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python ocr_gui_app.py
   ```

## How to Use

### Step 1: Configure Character Sets
1. Use the **"Character Set Selection"** checkboxes to choose which character types to include
2. Available options: Numbers (0-9), Uppercase (A-Z), Lowercase (a-z), Special (!@#$%)
3. Character count updates automatically as you make selections
4. **Process**: Creates filtered training dataset with only selected characters

### Step 2: Configure Training Parameters (Optional)
1. Click the **"Training Parameters"** tab to customize network settings
2. **Key Parameters**:
   - **Hidden Layer 1**: Default 64 neurons (adjustable 32-256)
   - **Hidden Layer 2**: Default 32 neurons (adjustable 32-128)
   - **Dropout Rate**: Default 0.2 (adjustable 0.0-0.8)
   - **Learning Rate**: Default 0.001 (adjustable 0.0001-0.1)
   - **Epochs**: Default 100 (adjustable 10-1000)
3. Save your configuration using **"Save Parameters"** for future use

### Step 3: Train the Model
1. Click the **"Train Model"** button in the Training section
2. **Training Process**:
   - Generates training data with noise variations
   - Builds neural network with selected architecture
   - Trains using Adam optimizer with configured parameters
   - Validates performance on held-out data
3. Monitor progress in "Training Progress" tab and system logs

### Step 4: Draw a Character
1. Use the 5x7 grid in the "Draw Character" section
2. **Drawing Process**: Click cells to toggle between 0 (white) and 1 (black)
3. Each cell represents one pixel in the character pattern
4. Draw any character from the selected character sets

### Step 5: Recognize the Character
1. Click the **"Recognize"** button after drawing
2. **Recognition Process**:
   - Flattens 5x7 grid to 35-element input vector
   - Passes through trained neural network
   - Filters predictions to only selected character sets
   - Returns character with highest confidence
3. View results in "Recognition Results" tab with ASCII code and confidence
4. See detailed network processing in "Network Processing" tab

## Network Architecture

The ANN uses a simplified architecture optimized for small datasets with configurable parameters:

```
Input Layer:     35 neurons (5x7 flattened matrix)
    ↓
Hidden Layer 1:  Configurable neurons (default: 64, ReLU activation)
    ↓
Dropout:         Configurable rate (default: 0.2) for regularization
    ↓
Hidden Layer 2:  Configurable neurons (default: 32, ReLU activation)
    ↓
Output Layer:    Variable neurons (Softmax activation - one per selected character)
```

**Architecture Benefits:**
- **Simplified Design**: Only 2 hidden layers to prevent overfitting with limited training data
- **Configurable Parameters**: Adjust layer sizes and dropout rate via GUI
- **Efficient Training**: Faster convergence with smaller architecture
- **Better Generalization**: Reduced complexity improves performance on small datasets

## Visualization Features

### Training Progress Tab
- **Accuracy Plot**: Shows training and validation accuracy over epochs
- **Loss Plot**: Displays training and validation loss curves
- **Real-time Updates**: Plots update as training progresses

### Network Processing Tab
- **Input Pattern**: Visual representation of your 5x7 input
- **Hidden Layer Activations**: Heatmaps showing neuron activations
- **Output Probabilities**: Bar chart with confidence scores and ASCII codes highlighted
- **Color-coded Visualizations**: Different color schemes for each layer

## Character Set Selection

Choose from multiple character sets to customize your training and recognition:

### Available Character Sets
- **Numbers (10 characters)**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Uppercase Letters (26 characters)**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Lowercase Letters (26 characters)**: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
- **Special Characters (25+ characters)**: !, ?, ., ,, ;, :, ', ", +, -, *, /, =, @, #, $, %, &, (, ), [, ], \\, _, space

### Usage Examples
- **Numbers Only**: Select only "Numbers (0-9)" for digit recognition
- **Letters Only**: Select "Uppercase (A-Z)" and "Lowercase (a-z)" for alphabetic recognition
- **Custom Mix**: Select any combination based on your specific needs

Each character is represented as a 5x7 binary matrix where:
- **1 (Black cell)**: Character pixel  
- **0 (White cell)**: Background pixel

## Technical Details

### Model Training Process
- **Optimizer**: Adam with configurable learning rate (default: 0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: Configurable (1-64, default: 8)
- **Epochs**: Configurable (10-1000, default: 100)
- **Validation Split**: Configurable (10%-50%, default: 20%)
- **Architecture**: 2-layer hidden network with configurable neuron counts
- **Regularization**: Dropout after first hidden layer to prevent overfitting
- **Training Data**: Generated from character patterns with noise variations for robustness

### Data Processing Pipeline
1. **Character Selection**: Filter training data based on selected character sets (Numbers, Uppercase, Lowercase, Special)
2. **Pattern Flattening**: Convert 5x7 matrix patterns to 35-element vectors
3. **Noise Augmentation**: Generate 2 noisy variants per character for robustness
4. **Label Encoding**: Convert character labels to numerical format for training
5. **Train/Validation Split**: Automatic data splitting based on configured ratio
6. **Batch Processing**: Process data in configurable batch sizes during training

### Visualization Technology
- **GUI Framework**: Tkinter with responsive design
- **Plotting**: Matplotlib with Tkinter integration
- **Real-time Updates**: Threading for non-blocking operations

## File Structure

```
ocr/
├── ocr_gui_app.py           # Main GUI application with training parameters
├── ocr_ann_system.py        # Core OCR neural network class (cleaned)
├── character_patterns.py    # Comprehensive 5x7 character pattern library
├── comprehensive_demo.py    # Complete demonstration with all characters
├── example_training_config.json  # Example training parameter configuration
├── requirements.txt         # Python dependencies
├── 5x7 Optical character.xlsx  # Sample Excel training data
└── README.md               # This file
```

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Tkinter (usually included with Python)
- OpenPyXL (for Excel support)

## Troubleshooting

### Common Issues

1. **"Model not trained" error**
   - Solution: Click "Train Model" before attempting recognition

2. **Blank visualization**
   - Solution: Draw a character and click "Recognize" first

3. **Training fails**
   - Check that TensorFlow is properly installed
   - Ensure sufficient memory is available
   - Verify at least one character set is selected

