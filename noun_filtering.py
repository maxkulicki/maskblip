import torch
import torch.nn as nn
from lavis.models import load_model
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
class LabelDataset(Dataset):
    def __init__(self, dataset_dir, embedding_model, n_samples=None, ):
        dataset = json.load(open(dataset_dir))
        self.word_lists = []
        self.labels = []
        self.embedding_lists = []
        for key in dataset:
            words = dataset[key]['original']
            self.word_lists.append(words)
            self.labels.append(dataset[key]['assignment'])
            self.embedding_lists.append(get_phrase_embeddings(embedding_model, words))
    def __len__(self):
        # this should return the size of the dataset
        return len(self.word_lists)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        #word_list = self.word_lists[idx]
        embedding_list = self.embedding_lists[idx]
        labels = self.labels[idx]
        return embedding_list, labels

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
    losses = []
    accuracies = []
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
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs.squeeze())
            preds = torch.round(outputs)  # Round to 0 or 1 for binary classification
            #print(outputs, labels)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data) / len(labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        losses.append(epoch_loss)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        accuracies.append(epoch_acc.cpu().detach().numpy())
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model, losses, accuracies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d_model = 768
nhead = 10
num_layers = 8
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

dataset = LabelDataset("noun_filtering_data.json", encoder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print("Dataset size: {}".format(len(dataset)))
# Suppose you have a batch of 8 sequences, each containing 10 word embeddings of size 300.
#src = torch.rand(10, 8, 300)

model, losses, accuracies = train(model, torch.nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=0.001), dataloader, device, num_epochs=50)

axs, fig = plt.subplots(2)
fig[0].plot(losses)
fig[0].set_title("Loss")
fig[1].plot(accuracies)
fig[1].set_title("Accuracy")
plt.show()


