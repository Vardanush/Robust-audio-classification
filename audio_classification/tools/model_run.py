"""
Classification training routine
"""

from audio_classification.data.urbansound8k import UrbanSoundDataset
import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
from audio_classification.model.deepcnn import lit_m18, lit_m11
from tqdm import tqdm
import datetime

__all__ = ["run"]

csv_path = '/nfs/students/winter-term-2020/project-1/datasets/UrbanSound8K/metadata/UrbanSound8K.csv'
file_path = '/nfs/students/winter-term-2020/project-1/datasets/UrbanSound8K/audio/'
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu


def get_dataloader():
    # downsampling to 8kHz
    transform = torchaudio.transforms.Resample(44100, 8000)

    train_set = UrbanSoundDataset(csv_path, file_path, range(1, 9), transform=transform)  # setting folder 9 as test_set
    val_set = UrbanSoundDataset(csv_path, file_path, [10], transform=transform)
    print("Train set size: " + str(len(train_set)))
    print("Validation set size: " + str(len(val_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, **kwargs)

    return train_loader, val_loader


def validate(model, train_loader, val_loader):
    model.eval()
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for clips, labels in loader:
                clips = clips.to(device=device)
                labels = labels.to(device=device)
                outputs = model(clips)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_train = 0.0
        pbar = tqdm(train_loader)
        for clips, labels in pbar:
            clips = clips.to(device=device)
            labels = labels.to(device=device)
            outputs = model(clips)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

            pbar.set_description("Current training loss " + str(loss.item()))

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))
        if epoch == 1 or epoch % 5 == 0:
            validate(model, train_loader, val_loader)


def run():
    train_loader, val_loader = get_dataloader()
    model = m18(num_classes=10).to(device=device)  # change to m11 or resnet1d
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0000001)
    loss_fn = nn.CrossEntropyLoss()
    training_loop(
        n_epochs=400,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
    )


# todo: general test loop
