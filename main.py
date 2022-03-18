import librosa
import torch
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path=None, input_len=16, mode='train', sr=16000, input_dim=63):
        self.input_len = input_len
        self.mode = mode
        gt = pd.read_csv(label_path)
        self.mfcc_samples = torch.zeros((gt.shape[0]+input_len, input_dim), dtype=torch.float)

        print("loading data now")

        for i, path in enumerate(gt['track']):
            y, sr = librosa.load(os.path.join(data_path, path), sr=32000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1).squeeze()
            mfcc_norm = librosa.util.normalize(mfcc)
            mfcc_norm = torch.from_numpy(mfcc_norm).float()
            self.mfcc_samples[i+input_len] = mfcc_norm

        if mode == 'train':        
            self.target = torch.stack([torch.tensor(score) for score in gt['score']])

    def __getitem__(self, index):
        output = self.mfcc_samples[index:index+self.input_len]
        if self.mode == 'train':
            target = self.target[index]
            return  output, target
        else:
            return output

    def __len__(self):
        return len(self.mfcc_samples)-self.input_len

class GRU_regression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, _x):
        x, h_n = self.gru(_x)
        s, b, h = x.shape
        x = x.reshape(s*b, h)
        x = self.fc(x)
        x = x.reshape(s, b, 1)
        return x

def train(args):
    dataset_train = Dataset('audios/clips','train.csv', input_len=args.input_len, \
                            mode='train', sr=args.sr, input_dim=args.input_dim)
        
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    loss_function = nn.MSELoss()
    model = GRU_regression(args.input_dim, hidden_size=args.hidden_dim, num_layers=1)

    opt = optim.SGD(model.parameters(), lr=0.2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    model.train()

    for epoch in range(args.epochs):
        losses = 0
        for x, t in train_loader:
            opt.zero_grad()
            x = x.squeeze().reshape(args.input_len, 1, args.input_dim).to(device)
            
            t = t.reshape(-1, 1, 1).to(device)
            out = model(x)
            loss = loss_function(out[-1].sigmoid(), t)
            loss.backward()
            opt.step()
            losses += loss.item()
        print('loss:', losses)

    torch.save(model.state_dict(), 'model_final.pth')

def test(args):
    model = GRU_regression(args.input_dim, hidden_size=args.hidden_dim, num_layers=1)
    model.load_state_dict(torch.load('model_final.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset_test = Dataset('audios/clips','test.csv', input_len=args.input_len, mode='test', \
                            sr=args.sr, input_dim=args.input_dim)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print('output test result:')
    for x in test_loader:
        x = x.squeeze().reshape(args.input_len, 1, args.input_dim).to(device)
        out = model(x)
        out = out[-1].sigmoid()
        print(out.item())

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='which GPU')
    parser.add_argument('--no_cuda', action='store_true', help='use CPU')
    parser.add_argument('--epochs', default=1000, help='training time')
    parser.add_argument('--input_len', default=32, help='GRU time step')
    parser.add_argument('--input_dim', default=313, help='input dimension')
    parser.add_argument('--hidden_dim', default=32, help='hidden dimension')
    parser.add_argument('--sr', default=16000, help='sample rate')
    parser.add_argument('--mode', default='train', help='chose training or testing')
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main()