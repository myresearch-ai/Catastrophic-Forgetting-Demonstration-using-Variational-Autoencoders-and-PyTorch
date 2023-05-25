# Catastrophic Forgetting in Neural Networks: A Variational Autoencoder (VAE) Case Study

This repository presents a PyTorch-based implementation demonstrating the phenomenon of catastrophic forgetting in Variational Autoencoders (VAEs) using MNIST and FashionMNIST datasets.

## Introduction

Catastrophic forgetting, also known as catastrophic interference, is a phenomenon observed in artificial neural networks where the network tends to abruptly forget previously learned information upon learning new data. This poses a significant challenge in the pursuit of continual or lifelong machine learning, where the objective is for models to learn from a continuous stream of data, building on past knowledge and adapting to new tasks - much like how humans learn.

In this repository, we demonstrate catastrophic forgetting by training a VAE first on the MNIST dataset, and subsequently on the FashionMNIST dataset. We observe that as the VAE learns to generate images of clothing items (FashionMNIST), it starts to forget how to generate images of handwritten digits (MNIST).

Repository Structure

The repository is structured as follows:

	•	VAE.py: This file contains the architecture of the Variational Autoencoder.
	•	loss.py: This file defines the loss function used for training the VAE.
	•	train.py: This script contains the function for training the VAE model.
	•	plot.py: This script contains the function for plotting the reconstruction images.
	•	main.py: This is the main script where everything is tied together, performing the experiment.
	•	requirements.txt: This file lists the Python dependencies required to run the code.

How to Run

To run the experiment, simply execute the main.py script:

python main.py

This will automatically download the MNIST and FashionMNIST datasets, train the VAE model on both, and save the reconstruction plots. Please ensure you have all necessary dependencies installed (i.e., PyTorch and torchvision).

## Requirements

To run the scripts, you will need the following:

- Python 3.7 or higher
- PyTorch 1.8.1 or higher
- torchvision 0.9.1 or higher
- matplotlib

You can install the dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

Train the model:

```
python main.py
```

Plot the results:

```
python plot.py
```

## Results

The results are demonstrated through a plot of the reconstruction loss over time for both the MNIST and FashionMNIST datasets. As the model is trained on the FashionMNIST dataset, the reconstruction loss for the MNIST dataset increases, illustrating the occurrence of catastrophic forgetting.

## Future Work

Catastrophic forgetting remains an open problem in the field of AI, and solving it is a critical step towards creating AI models capable of continuous learning. Various techniques have been proposed to alleviate this problem, including methods like Elastic Weight Consolidation (EWC) and Progressive Neural Networks (PNNs). Implementing these strategies in the context of this experiment will be the direction of future work.

## Contributions

Contributions to improve this demonstration or to implement solutions to catastrophic forgetting are very welcome. Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the terms of the MIT license.
