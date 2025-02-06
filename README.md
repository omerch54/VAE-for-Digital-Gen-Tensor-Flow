**VAE and CVAE Implementation for MNIST**  

## **Overview**  
This project implements **Variational Autoencoders (VAE)** and **Conditional Variational Autoencoders (CVAE)** using TensorFlow. The models are trained on the **MNIST dataset** to generate and reconstruct handwritten digits. The system supports both **unsupervised learning (VAE)** and **conditional image generation (CVAE)** by incorporating class labels into the latent space.

---

## **Features**  
- **Variational Autoencoder (VAE)**
  - Learns a latent representation of MNIST digits.
  - Generates new digits from the learned latent space.
  - Supports latent space interpolation for visualization.

- **Conditional Variational Autoencoder (CVAE)**
  - Uses class labels to guide image generation.
  - Generates digits conditioned on specific labels.

- **MNIST Data Preprocessing**
  - Normalizes images to the [0,1] range.
  - Converts class labels to one-hot encodings for CVAE.

- **Training and Evaluation**
  - Trains VAE/CVAE with customizable parameters.
  - Uses **reparameterization trick** for latent space sampling.
  - Computes **VAE loss** including reconstruction loss and KL divergence.

- **Visualization**
  - Generates images from the latent space.
  - Performs latent space interpolation to explore feature transitions.
  - Saves generated images to an output directory.

---

## **Installation & Dependencies**  
Ensure you have Python installed and run the following command to install required dependencies:  
```bash
pip install tensorflow numpy matplotlib tqdm argparse
```

---

## **Usage**  

### **1. Train the Model**  
To train a **VAE**:  
```bash
python assignment.py --num_epochs 10 --batch_size 128 --latent_size 15
```

To train a **CVAE**:  
```bash
python assignment.py --is_cvae --num_epochs 10 --batch_size 128 --latent_size 15
```

### **2. Load Pretrained Weights**
To train using saved weights, add the `--load_weights` flag:
```bash
python assignment.py --is_cvae --load_weights
```

### **3. Customize Hyperparameters**  
Modify training parameters via command-line arguments:
- `--batch_size`: Batch size for training (default: 128).
- `--num_epochs`: Number of training epochs (default: 10).
- `--latent_size`: Dimensionality of the latent space (default: 15).
- `--learning_rate`: Learning rate for optimizer (default: 1e-3).
- `--is_cvae`: Use Conditional VAE instead of standard VAE.

---

## **Project Structure**
```
├── assignment.py       # Main script for training and evaluation
├── vae.py              # Contains VAE and CVAE model definitions
├── model_ckpts/        # Stores trained model weights
├── outputs/            # Stores generated images
└── requirements.txt    # Lists required dependencies
```

---

## **Key Components**  

### **1. Data Processing**  
- Loads MNIST dataset and normalizes it.  
- Converts images to tensors and applies batching.  

### **2. Model Architecture**  
- **VAE**: Encoder learns latent space, Decoder reconstructs images.  
- **CVAE**: Adds class labels as an additional condition in the latent space.  

### **3. Training Loop**  
- Uses **Adam optimizer** to minimize VAE loss.  
- Applies **KL divergence and reconstruction loss**.  
- Supports training both **VAE** and **CVAE** via a single pipeline.  

### **4. Visualization**  
- Generates 10 images from random latent vectors.  
- Performs **latent space interpolation** to visualize smooth transformations between digits.  
- Saves generated images to `outputs/` for analysis.  

---

## **Saving and Loading Models**
The trained models are stored in the `model_ckpts/` directory:
- **VAE Weights**: `model_ckpts/vae/vae`
- **CVAE Weights**: `model_ckpts/cvae/cvae`

To save model weights after training:
```python
save_model_weights(model, args)
```

To load weights before inference:
```python
model = load_weights(model)
```

---

## **Results & Expected Output**  
- **VAE Generated Images:** Unsupervised generation of handwritten digits.  
- **CVAE Generated Images:** Controlled digit generation based on input class labels.  
- **Latent Space Interpolation:** Smooth transition between different digit styles.  

---

## **Future Improvements**  
- Train on more complex datasets beyond MNIST.  
- Implement **beta-VAE** for better disentangled representations.  
- Optimize latent space exploration with **semi-supervised learning**.  

---

## **Contributors**  
- **Omer Chaudhry**  

