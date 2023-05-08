# Transformer using PyTorch

This project is an experiment in text generation using PyTorch. The initial development of a simple bigram model with an embedding layer progressed to creating a complete transformer decoder model using character generation. The dataset used is Shakespeare's scripts. Individual components of the transformer were developed in different notebook files. 

## Project URL
The code for this project can be found on [GitHub](https://github.com/ritikjain51/transformers-pytorch).

## Files
- `bigram_model.ipynb`: Jupyter notebook containing the code for the simple bigram model
- `transformer_decoder.ipynb`: Jupyter notebook containing the code for the transformer decoder model using character generation
- `components/`: Directory containing Jupyter notebooks for individual components of the transformer

## Dataset
The dataset used in this project is Shakespeare's scripts, which can be found in the `data/` directory.

## Requirements
- PyTorch
- NumPy
- Matplotlib

## Usage
1. Clone the repository: `git clone https://github.com/ritikjain51/transformers-pytorch.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebooks in the order specified above to train and generate text using the bigram model and transformer decoder.

## Acknowledgements
- This project was inspired by the Transformer architecture proposed in the paper "Attention is All You Need" by Vaswani et al.
- The Shakespeare dataset used in this project was obtained from [Project Gutenberg](https://www.gutenberg.org/).
