import torch
import torch.nn as nn
from lavis.models import load_model

class SelfAttentionBinaryClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.5):
        super(SelfAttentionBinaryClassifier, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers
        )

        # Use a single output unit with a sigmoid function for binary classification.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, src):
        output = self.self_attention(src)
        output = self.classifier(output)
        return output.squeeze(-1)

def get_phrase_embeddings(model, phrase_list):
    phrase_embeddings = []
    for phrase in phrase_list:
        embedding = model.extract_features({'text_input': phrase}, mode="text")['text_embeds'][:,0,:]
        phrase_embeddings.append(embedding)
    return torch.stack(phrase_embeddings)


def train(model, criterion, optimizer, dataloader, device, num_epochs=25):
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloader:
            # inputs is your batch of sequences of word embeddings,
            # and labels is the batch of target labels.
            # You might need to adjust the shape of these depending on how your DataLoader is set up.

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            preds = torch.round(outputs)  # Round to 0 or 1 for binary classification
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 768
nhead = 2
num_layers = 2
phrases =[
        "garbage filled street",
        "animal",
        "concrete",
        "dirty building",
        "cow",
        "dirt area",
        "garbage",
        "street",
        "background",
        "garbage truck",
        "front",
        "some trash",
        "small cow",
        "top",
        "bull",
        "garbagester pile",
        "wall",
        "dirt",
        "white cow",
        "pile",
        "edge"
    ]
encoder = load_model("blip_feature_extractor", "base")
phrase_embeddings = get_phrase_embeddings(encoder, phrases)
model = SelfAttentionBinaryClassifier(d_model, nhead, num_layers)

# Suppose you have a batch of 8 sequences, each containing 10 word embeddings of size 300.
#src = torch.rand(10, 8, 300)

output = model(phrase_embeddings)

train(model, torch.nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=0.001), dataloader, device, num_epochs=25)

print(output)