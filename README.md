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

### Step 2: Configure Training Parameters (Optional)
1. Click the **"Training Parameters"** tab to customize network settings
2. Adjust epochs, learning rate, network architecture, and dropout rates
3. Save your configuration using **"Save Parameters"** for future use

### Step 3: Train the Model
1. Click the **"Train Model"** button in the Training section
2. Monitor detailed training logs in the System Log section
3. Watch training progress in the "Training Progress" tab

### Step 4: Draw a Character
1. Use the 5x7 grid in the "Draw Character" section
2. Click cells to toggle them black/white
3. Draw any character from the selected character sets

### Step 5: Recognize the Character
1. Click the **"Recognize"** button
2. View results in the "Recognition Results" section with highlighted ASCII code
3. Switch to the "Network Processing" tab to see detailed neural network processing with ASCII codes

### Step 6: Import Excel Data (Optional)
1. Click **"Import Excel File"** to load additional training data
2. Supported formats: .xlsx, .xls files

## Network Architecture

The ANN uses the following configurable architecture:

```
Input Layer:     35 neurons (5x7 flattened matrix)
    ↓
Hidden Layer 1:  Configurable neurons (default: 128, ReLU activation) + Dropout
    ↓
Hidden Layer 2:  Configurable neurons (default: 96, ReLU activation) + Dropout
    ↓
Hidden Layer 3:  Configurable neurons (default: 64, ReLU activation) + Dropout
    ↓
Hidden Layer 4:  Configurable neurons (default: 32, ReLU activation)
    ↓
Output Layer:    Variable neurons (Softmax activation - one per selected character)
```

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

### Model Training
- **Optimizer**: Adam (configurable learning rate)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: Configurable (1-64, default: 8)
- **Epochs**: Configurable (10-1000, default: 100)
- **Validation Split**: Configurable (10%-50%, default: 20%)
- **Architecture**: Fully customizable layer sizes and dropout rates

### Data Augmentation
- **Noise Injection**: Random bit flipping for robustness
- **Variation Generation**: 2-3 noisy variants per base pattern
- **Filtered Training**: Only selected character sets included

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

