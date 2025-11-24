# Make More Part 2 - Character-Level Language Model for Farsi Names

A PyTorch-based character-level language model that generates Farsi (Persian) names using a deep neural network. This project implements a custom neural network architecture from scratch, including embedding layers, batch normalization, and multi-layer perceptrons.

## 🎯 Overview

This project implements a character-level language model that learns to generate Farsi names by training on a dataset of Persian names. The model uses a context window approach where it predicts the next character based on the previous characters, making it capable of generating new, plausible Farsi names.

The implementation includes:
- Custom neural network layers built from scratch
- Batch normalization for stable training
- Character embeddings for learning character relationships
- Comprehensive training loop with learning rate scheduling
- Visualization tools for monitoring training progress

## ✨ Features

- **Character-Level Generation**: Generates names character by character using a context window
- **Custom Layer Implementation**: Built-in Linear, BatchNorm, and Tanh layers
- **GPU Support**: Automatic CUDA detection and device management
- **Batch Normalization**: Custom BatchNorm implementation for training stability
- **Training Visualizations**: Activation distributions, gradient histograms, and update ratio tracking
- **Flexible Architecture**: Configurable embedding dimensions, hidden layers, and network depth
- **Reproducible Results**: Seed-based random number generation for consistent results

## 🏗️ Architecture

The model consists of:

1. **Character Embedding Layer**: Maps each character to a dense vector representation
2. **Multi-Layer Perceptron (MLP)**: Stack of linear layers with batch normalization and tanh activations
3. **Output Layer**: Produces logits over the vocabulary for next character prediction

### Network Structure

```
Input (context window) → Embedding → Flatten → 
[Linear → BatchNorm → Tanh] × N → 
Linear → BatchNorm → Output (vocab_size)
```

## 📦 Requirements

- Python 3.7+
- PyTorch (with CUDA support optional but recommended)
- NumPy
- Matplotlib

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "Make More Part2"
```

2. Install required packages:
```bash
pip install torch matplotlib numpy
```

Or create a `requirements.txt` file with:
```
torch>=2.0.0
matplotlib>=3.5.0
numpy>=1.21.0
```

Then install:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Training

Run the main script to train the model:

```bash
python MakeMoreP2.py
```

The script will:
1. Load the Farsi names dataset
2. Split data into training, validation, and test sets
3. Initialize the neural network
4. Train the model for the specified number of steps
5. Generate sample names from the trained model

### Configuration

You can modify hyperparameters in the script:

```python
# Hyperparameters
n_emb = 10              # Embedding dimension
n_hidden = 300          # Hidden layer size
block_size = 3          # Context window size
max_steps = 200000       # Training steps
batch_size = 32         # Batch size
initial_lr = 0.1        # Initial learning rate
```

### Generating Names

After training, the model will automatically generate 10 sample names. To generate more names, modify the loop in the sampling section:

```python
for _ in range(10):  # Change 10 to desired number
    # ... generation code ...
```

## 🔬 Model Architecture Details

### Custom Layers

#### Linear Layer
- Implements a fully connected layer with optional bias
- Supports Xavier/Glorot initialization
- Configurable gain for weight initialization

#### BatchNorm (BenchNorm)
- Custom batch normalization implementation
- Maintains running statistics for inference
- Configurable momentum for running statistics update

#### Tanh Activation
- Hyperbolic tangent activation function
- Provides non-linearity to the network

### Key Implementation Details

- **Context Window**: Uses a sliding window of 3 characters to predict the next character
- **Character Vocabulary**: Automatically builds vocabulary from the dataset
- **Special Token**: Uses '.' as a special end-of-name token
- **Weight Initialization**: Uses gain-based initialization (Xavier/Glorot) for stable training
- **Learning Rate Scheduling**: Implements step-based learning rate decay

## 📊 Training Details

### Data Split
- Training: 80% of the dataset
- Validation: 10% of the dataset
- Test: 10% of the dataset

### Training Process
1. Mini-batch gradient descent
2. Forward pass through the network
3. Cross-entropy loss calculation
4. Backward pass (backpropagation)
5. Parameter updates with weight decay

### Optimization
- Learning rate decay: Starts at 0.01, decays to 0.0001 after 25,000 steps
- Weight decay: L2 regularization with coefficient 0.000001
- Batch normalization: Applied after each linear layer (except output)

### Monitoring
The training loop prints loss every 1,000 steps:
```
   1000/200000, loss: 2.34567890
   2000/200000, loss: 2.12345678
   ...
```

## 📁 Project Structure

```
Make More Part2/
│
├── MakeMoreP2.py          # Main training and model script
├── Names_Farsi.txt         # Farsi names dataset
├── Model_params.pth        # Saved model parameters (if available)
└── README.md               # This file
```

## 📈 Visualization

The script includes several visualization features:

1. **Activation Distribution**: Histograms showing the distribution of activations at each layer
2. **Gradient Distribution**: Histograms of gradients flowing through the network
3. **Update Ratio**: Tracks the ratio of parameter updates to parameter values
4. **Saturation Analysis**: Monitors the percentage of saturated neurons (>0.97 or <-0.97)

These visualizations help diagnose training issues and understand model behavior.

## 🎓 Learning Objectives

This project demonstrates:
- Building neural networks from scratch
- Implementing batch normalization
- Character-level language modeling
- Training deep networks with proper initialization
- Monitoring and visualizing training dynamics

## 🔍 Technical Notes

- The model uses a context window of 3 characters (configurable via `block_size`)
- Character embeddings are learned during training
- The network depth can be adjusted by modifying the `layers` list
- GPU acceleration is automatically used if available
- All random operations use fixed seeds for reproducibility

## 🐛 Known Issues / Future Improvements

- The training loop includes a `break` statement that stops training early (line 214, 422) - remove for full training
- Some visualization code is commented out - uncomment to enable additional plots
- Consider adding model checkpointing for saving/loading trained models
- Could benefit from early stopping based on validation loss

## 📝 Notes

- The dataset path (`Names_Farsi.txt`) is currently hardcoded to a relative path. Ensure the file is in the correct location relative to the script.
- The model is designed for Farsi/Persian names but can be adapted for other character-level tasks by changing the dataset.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

---

**Note**: This is an educational project demonstrating neural network implementation from scratch. For production use, consider using established frameworks like PyTorch's built-in layers and optimizers.



