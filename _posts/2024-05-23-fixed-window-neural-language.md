---
layout: post
title: Fixed Window Neural Language
date: 2024-05-23 23:02 -0500
categories:
- Tech
- Natural Language Processing
tags:
- tutorial
- nlp
---

![example](/assets/img/fixed_win_nl_1.png){: width=auto height=auto }

*Implementing a simple fixed window Neural Language Model that predicts the third word in a sentence based on the first two words*

## Libs

```python
# importing libs
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
```
## Set up

```python
# This is our training set.
# our prediction is as good as our data.
corpus = [
    "John adores dogs",
    "Emma paints beautifully",
    "NLP is fascinating",
    "They cherish books",
    "Tyson is fast"
]

# defining our tokenization function
def tokenize(sentence):
    # we need to lowercase the sentence
    # and split it into words
    return sentence.lower().split()

# tokenize each sentence in our corpus
tokenized_corpus = [tokenize(sentence) for sentence in corpus]
print("tokenized_corpus =")
print(tokenized_corpus)
```

```output
tokenized_corpus = [['john', 'adores', 'dogs'], 
                    ['emma', 'paints', 'beautifully'], 
                    ['nlp', 'is', 'fascinating'], 
                    ['they', 'cherish', 'books'],
                    ['tyson', 'is', 'fast']]
```

```python
# define a set to contain all
# unique words in our corpus
vocab = set()

# for each tokenized sentence,
# update our vocab with any new words
for sentence in tokenized_corpus:
    vocab.update(sentence)

# create a dict that maps unique 
# words to unique numbers
word_to_index = {word: i for i, word in enumerate(sorted(vocab))}
word_to_index

print("word_to_index = ")
print(word_to_index)
```

```output
word_to_index = { 'adores': 0, 
                  'beautifully': 1, 
                  'books': 2, 
                  'cherish': 3, 
                  'dogs': 4, 
                  'emma': 5, 
                  'fascinating': 6, 
                  'fast': 7, 
                  'is': 8, 
                  'john': 9, 
                  'nlp': 10, 
                  'paints': 11, 
                  'they': 12, 
                  'tyson': 13
                }
```

## Transform

```python
def prepare_training_data(tokenized_corpus, word_to_index):
    """
    Args:
        tokenized_corpus: List[List[str]].
          --> sentences (n) by words (m)
        word_to_index: dict{str: int}.
 
    Returns:
        input_pairs: List[List[int]]
          --> sentences (n) by words except target (m-1)
        target_words: List[int])
          --> target word for each sentence (n)
    """
    input_pairs = []
    target_words = []

    # Iterate through the tokenized corpus
    for sentence in tokenized_corpus:
      # the first two are our input
      input_pair = (word_to_index[sentence[0]], word_to_index[sentence[1]])
      # the last word is our target
      target_word = word_to_index[sentence[-1]]

      # append
      input_pairs.append(input_pair)
      target_words.append(target_word)

    # Convert to tensors
    input_pairs = torch.tensor(input_pairs, dtype=torch.long)
    target_words = torch.tensor(target_words, dtype=torch.long)

    return input_pairs, target_words

# Prepare the training data
input_pairs, target_words = prepare_training_data(tokenized_corpus, word_to_index)
print(input_pairs, target_words)
```
```output
input_pairs = tensor([[ 9,  0],
                      [ 5, 11],
                      [10,  8],
                      [12,  3],
                      [13,  8]]) 

target_words = tensor([4, 1, 6, 2, 7])
```
> **Note**: These numbers map to words based on our `word_to_index` mapping

## Train

```python
# Define the neural network model
class FixedWindowNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(FixedWindowNLM, self).__init__()

        # Embedding layer;
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Hidden layer (linear)
        self.hidden = nn.Linear(embedding_dim * 2, hidden_dim)

        # Activation function (ReLU)
        self.relu = nn.ReLU()

        # Output layer (linear)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_pair):
        # Extract embeddings for the input pair
        embedded = self.embedding(input_pair)

        # Flatten and concatenate the embeddings of the input pair
        concatenated = embedded.view(embedded.size(0), -1)

        # Pass through hidden layer
        hidden_output = self.hidden(concatenated)

        # Apply activation function
        activated = self.relu(hidden_output)

        # Pass through output layer
        output = self.output(activated)

        return output

# Define model parameters
vocab_size = len(word_to_index) # or len(vocab)
embedding_dim = 5
hidden_dim = 12

# Create the model
model = FixedWindowNLM(vocab_size, embedding_dim, hidden_dim)

# Display the model architecture
model
```
```output
FixedWindowNLM(
  (embedding): Embedding(14, 5)
  (hidden): Linear(in_features=10, out_features=12, bias=True)
  (relu): ReLU()
  (output): Linear(in_features=12, out_features=14, bias=True)
)
```
> **Note**: In `Embedding(14, 5)`, the 14 refers to the number of unique words we have in our corpus and the 5 refers to our choice of embedding size for each word.

```python
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Define the number of epochs
num_epochs = 150

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_pairs)

    # Compute loss
    loss = criterion(outputs, target_words)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for each epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training is complete!")
```
```output
Epoch [10/150], Loss: 2.0092
Epoch [20/150], Loss: 1.3288
Epoch [30/150], Loss: 0.7458
Epoch [40/150], Loss: 0.3690
Epoch [50/150], Loss: 0.1803
Epoch [60/150], Loss: 0.0978
Epoch [70/150], Loss: 0.0622
Epoch [80/150], Loss: 0.0442
Epoch [90/150], Loss: 0.0338
Epoch [100/150], Loss: 0.0270
Epoch [110/150], Loss: 0.0224
Epoch [120/150], Loss: 0.0190
Epoch [130/150], Loss: 0.0164
Epoch [140/150], Loss: 0.0144
Epoch [150/150], Loss: 0.0128
Training is complete!
```

## Viewing the matrices

```python
print(model.embedding.weight)
print(model.embedding.weight.shape)
```
```output
Parameter containing:
tensor([[ 0.2688, -1.2350,  1.8463,  3.0616, -0.5894],
        [ 0.1482, -0.3205,  0.0531,  0.0378, -1.1383],
        [-0.6544,  1.1692, -0.5270,  1.1148,  1.6814],
        [-1.1172, -0.4641, -0.1556, -0.6631,  0.1833],
        [ 1.6813, -0.9086, -1.5162, -1.0999,  0.7923],
        [-0.6985, -1.6411,  0.5234, -0.7266, -0.6397],
        [-0.4285, -0.9649,  1.9820,  0.4501,  0.5230],
        [-0.3438, -0.5655,  0.0078,  1.8435,  0.4603],
        [-0.1448,  1.4380, -0.7473,  0.1697,  1.5595],
        [ 0.0655, -0.8886,  1.0023, -0.0142,  0.3770],
        [-0.8534,  1.3914,  1.5175, -0.2543, -1.1140],
        [ 0.4273,  0.9039, -1.8467,  2.0521, -0.2817],
        [ 1.3245, -0.4529, -0.4798,  0.3977, -0.2333],
        [ 0.1975,  1.5445, -0.3989, -0.8718,  1.9489]], requires_grad=True)
torch.Size([14, 5])
```

```python
print(model.hidden.weight)
print(model.hidden.weight.shape)
```
```output
Parameter containing:
tensor([[-3.6710e-02, -6.4246e-02,  3.9706e-02, -3.4790e-01,  6.8597e-01,
         -1.7007e-01,  3.3097e-01,  1.6404e-02, -1.9970e-01,  3.4376e-01],
        [-4.8082e-02, -1.7911e-01,  2.5092e-01, -2.9444e-01,  3.6004e-01,
          6.6798e-02, -3.4206e-01, -5.0198e-02,  1.9028e-01,  2.5640e-01],
        [-2.2445e-01, -5.9653e-01, -2.4925e-01, -1.2773e-01,  4.7297e-02,
          7.0669e-02,  3.0357e-01, -6.3473e-01,  1.7252e-01, -3.0109e-01],
        [ 1.0412e-01, -3.2042e-01,  2.5425e-01,  2.7436e-01,  2.4392e-01,
         -1.6533e-01, -6.9823e-02,  8.8925e-01,  8.6078e-01, -5.5306e-03],
        [ 6.1508e-01, -3.2004e-01, -4.5080e-01,  1.6075e-01,  3.5395e-02,
         -5.3901e-01, -4.2496e-01,  6.2967e-02, -1.0264e-01, -2.1777e-01],
        [-1.5308e-01,  3.7319e-01, -1.5164e-01, -3.7946e-01,  4.8048e-01,
         -3.5243e-02,  9.9701e-02, -1.7049e-01, -2.8865e-02,  3.5329e-01],
        [-4.1553e-01,  1.3424e-01,  2.6958e-01, -1.8012e-01, -5.1110e-01,
          4.0700e-02,  2.9680e-02, -4.1088e-01,  2.7892e-01,  2.0191e-01],
        [-5.5744e-01,  2.5725e-01,  6.5046e-01,  4.9939e-02, -7.5402e-01,
         -2.1389e-01,  1.2572e-01, -6.0297e-01,  3.9608e-01,  4.8535e-01],
        [ 2.9265e-01, -2.2741e-01,  4.9509e-02,  3.7863e-01, -2.5014e-01,
         -7.1770e-01, -1.8446e-01, -2.8271e-01, -5.0420e-01,  3.3103e-01],
        [ 2.0390e-01, -2.9077e-01,  2.6926e-01,  1.8451e-01,  6.2733e-02,
          5.9361e-03,  1.9562e-01,  1.4464e-01, -2.1479e-04,  2.8323e-01],
        [ 1.1318e-01,  8.5081e-02, -3.1291e-02,  2.1192e-02,  3.7771e-01,
          1.8995e-01,  3.8145e-01, -5.3564e-01,  4.1974e-01, -1.7279e-01],
        [-1.4464e-01,  7.1174e-01,  4.0156e-01,  6.1441e-03,  7.0387e-02,
         -9.1504e-02,  2.3581e-01, -1.8324e-01, -1.6341e-01,  4.3303e-01]],
       requires_grad=True)
torch.Size([12, 10])
```
```python
print(model.output.weight)
print(model.output.weight.shape)
```
```output
Parameter containing:
tensor([[-3.0356e-01,  1.9304e-01, -1.0269e-01, -2.1830e-01, -1.6539e-01,
          1.9917e-01,  1.0257e-01, -3.3066e-01,  5.9507e-02, -2.5905e-01,
         -3.0123e-02,  1.5670e-01],
        [ 1.3781e-01,  8.1654e-02,  9.6712e-01, -1.4224e-01, -2.0387e-01,
         -3.5698e-01,  5.2611e-01,  6.0250e-01, -3.8286e-01, -1.3487e-01,
          5.3536e-01, -5.0341e-01],
        [-3.4796e-01,  8.4468e-02,  1.6215e-01, -1.1580e-01,  9.9495e-01,
          9.6676e-02,  1.1415e-01, -2.7826e-01,  1.0677e+00,  5.6870e-02,
          1.0664e-01, -3.0830e-02],
        [ 1.2047e-01,  6.1328e-02, -1.4419e-01, -7.5545e-02, -1.5464e-01,
          2.3420e-02, -1.4633e-01, -1.4967e-01, -3.9336e-02, -1.3515e-01,
          1.4505e-01, -8.9117e-02],
        [ 9.4172e-02,  4.2784e-01, -2.0584e-01,  1.2413e+00,  1.0281e-01,
         -2.8040e-02, -2.3257e-01, -1.9368e-01, -1.7047e-01,  6.4686e-02,
         -4.8812e-04, -1.8709e-01],
        [-4.4395e-02, -7.0717e-03,  1.1692e-01,  1.8102e-01, -8.8736e-02,
          1.5119e-01, -1.7189e-01, -3.0413e-01, -6.7960e-02, -1.4524e-01,
         -2.9749e-01, -2.0616e-01],
        [-2.4073e-01, -1.3725e-01, -3.0749e-01,  6.7514e-02,  3.1825e-02,
          2.2228e-03,  5.2561e-01,  1.1133e+00,  1.8977e-01, -1.0051e-01,
         -4.8430e-02,  5.0951e-01],
        [ 6.7107e-01,  3.4710e-01,  1.6504e-01, -8.4050e-02, -2.0008e-01,
          7.8587e-01, -1.8245e-01, -2.1405e-01, -3.0980e-01, -1.5638e-01,
          5.5969e-01,  4.6355e-01],
...
        [-2.5938e-01, -1.0647e-01, -5.5407e-02, -4.2755e-02, -2.2404e-01,
         -1.2027e-01, -1.8324e-01,  5.0544e-02, -1.7049e-01, -2.8036e-01,
         -1.5554e-01, -1.8482e-01]], requires_grad=True)
torch.Size([14, 12])
```

## Predict

let's use this model ...

```python
# Convert the input to corresponding indices
input_words = ("john", "adores")
input_indices = torch.tensor([vocab_index[input_words[0]], vocab_index[input_words[1]]], dtype=torch.long)

# Use the model to predict the third index
with torch.no_grad():
    # Forward pass; unqueeze(0) takes it from [] to [[]]
    outputs = model(input_indices.unsqueeze(0))

    # Apply softmax to get the prediction probabilities
    probabilities = torch.softmax(outputs, dim=1)

    # we do not want to sort because we would lose the index
    # Get the predicted index with the highest probability
    predicted_index = torch.argmax(probabilities, dim=1).item()

    # Get the predicted word from the index
    predicted_word = list(vocab_index.keys())[list(vocab_index.values()).index(predicted_index)]

# Display the prediction and its probability
predicted_word, probabilities[0, predicted_index].item()
```
```output
('dogs', 0.9915106892585754)
```
The input `John adores` has been biased to pick `dogs` because that is all there is in our corpus. This demonstrates the need for a good corpus.
## Tweak learning rate
If you wish to test different `lr` values
```python
def train_and_predict(learning_rate):
    model = FixedWindowNLM(vocab_size, embedding_dim=5, hidden_dim=12)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 150
    for _ in range(num_epochs):

        outputs = model(input_pairs)
        loss = criterion(outputs, target_words)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    input_words = ("john", "adores")
    input_indices = torch.tensor([vocab_index[input_words[0]], vocab_index[input_words[1]]], dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_indices.unsqueeze(0))
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_word = list(vocab_index.keys())[list(vocab_index.values()).index(predicted_index)]

    # Return the predicted word and its probability
    return predicted_word, probabilities[0, predicted_index].item()

# Learning rates to test
learning_rates = [0.9, 0.5, 0.01, 0.0001]

# Iterate through learning rates and perform training and prediction
for lr in learning_rates:
    predicted_word, probability = train_and_predict(lr)
    print(f"Learning Rate: {lr}")
    print(f"Predicted Word: {predicted_word}")
    print(f"Prediction Probability: {probability:.4f}\n")
```
```output
Learning Rate: 0.9
Predicted Word: dogs
Prediction Probability: 0.9994

Learning Rate: 0.5
Predicted Word: dogs
Prediction Probability: 0.9990

Learning Rate: 0.01
Predicted Word: dogs
Prediction Probability: 0.1312

Learning Rate: 0.0001
Predicted Word: nlp
Prediction Probability: 0.0878
```