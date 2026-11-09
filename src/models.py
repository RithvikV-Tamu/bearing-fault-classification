"""Logistic regression (scikit-learn) and a small neural network (PyTorch).

Hyperparameters were fixed before any test results were looked at.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

RANDOM_SEED = 42
NUMBER_OF_CLASSES = 4

NETWORK_HIDDEN_SIZES = (256, 64)
NETWORK_DROPOUT = 0.3
NETWORK_EPOCHS = 40
NETWORK_BATCH_SIZE = 128
NETWORK_LEARNING_RATE = 1e-3
NETWORK_WEIGHT_DECAY = 1e-4


def train_logistic_regression(train_features, train_labels):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    model.fit(train_features, train_labels)
    return model


def predict_logistic_regression(model, test_features):
    return model.predict(test_features)


class SpectrumClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        first_hidden, second_hidden = NETWORK_HIDDEN_SIZES
        self.layers = nn.Sequential(
            nn.Linear(input_size, first_hidden),
            nn.ReLU(),
            nn.Dropout(NETWORK_DROPOUT),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(NETWORK_DROPOUT),
            nn.Linear(second_hidden, NUMBER_OF_CLASSES),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def set_random_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_neural_network(train_features, train_labels):
    set_random_seeds()
    scaler = StandardScaler().fit(train_features)
    inputs = torch.tensor(scaler.transform(train_features), dtype=torch.float32)
    targets = torch.tensor(train_labels, dtype=torch.long)

    class_counts = np.bincount(train_labels, minlength=NUMBER_OF_CLASSES)
    class_weights = len(train_labels) / (NUMBER_OF_CLASSES * class_counts)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))

    network = SpectrumClassifier(inputs.shape[1])
    optimizer = torch.optim.Adam(
        network.parameters(), lr=NETWORK_LEARNING_RATE, weight_decay=NETWORK_WEIGHT_DECAY
    )
    shuffle_generator = torch.Generator().manual_seed(RANDOM_SEED)

    network.train()
    for epoch in range(NETWORK_EPOCHS):
        order = torch.randperm(len(inputs), generator=shuffle_generator)
        for start in range(0, len(inputs), NETWORK_BATCH_SIZE):
            batch_indices = order[start:start + NETWORK_BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(network(inputs[batch_indices]), targets[batch_indices])
            loss.backward()
            optimizer.step()

    return {"network": network, "scaler": scaler}


def predict_neural_network(model, test_features):
    network = model["network"]
    network.eval()
    inputs = torch.tensor(model["scaler"].transform(test_features), dtype=torch.float32)
    with torch.no_grad():
        return network(inputs).argmax(dim=1).numpy()
