import os
import argparse
import torch
import torchaudio
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import librosa
import speechbrain
from tqdm import tqdm
import pandas as pd
#figas


   
class MosPredictor(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.mean_net_conv = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3,3), padding = (1,1), stride=(1,3)),
            nn.Dropout(0.3),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        self.relu_ = nn.ReLU()
        self.sigmoid_ = nn.Sigmoid()
        
        self.ssl_features = 1280
        self.dim_layer = nn.Linear(self.ssl_features, 512)

        self.mean_net_rnn = nn.LSTM(input_size = 512, hidden_size = 128, num_layers = 1, batch_first = True, bidirectional = True)
        self.mean_net_dnn = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )        

        self.sinc = speechbrain.nnet.CNN.SincConv(in_channels=1, out_channels=257, kernel_size=251, stride=256, sample_rate=16000)
        self.att_output_layer_quality = nn.MultiheadAttention(128, num_heads=8)                
        self.output_layer_quality = nn.Linear(128, 1)
        self.qualaverage_score = nn.AdaptiveAvgPool1d(1)  
     
        self.att_output_layer_intell = nn.MultiheadAttention(128, num_heads=8)           
        self.output_layer_intell = nn.Linear(128, 1)
        self.intellaverage_score = nn.AdaptiveAvgPool1d(1)  
                       
        self.att_output_layer_stoi= nn.MultiheadAttention(128, num_heads=8)          
        self.output_layer_stoi = nn.Linear(128, 1)        
        self.stoiaverage_score = nn.AdaptiveAvgPool1d(1) 

    def new_method(self):
        self.sin_conv 
                
    def forward(self, wav, lps, whisper):
        #SSL Features
        wav_ = wav.squeeze(1)  ## [batches, audio_len]
        ssl_feat_red = self.dim_layer(whisper.squeeze(1))
        ssl_feat_red = self.relu_(ssl_feat_red)
 
        #PS Features
        sinc_feat=self.sinc(wav.squeeze(1))
        unsq_sinc =  torch.unsqueeze(sinc_feat, axis=1)
        concat_lps_sinc = torch.cat((lps,unsq_sinc), axis=2)
        cnn_out = self.mean_net_conv(concat_lps_sinc)
        batch = concat_lps_sinc.shape[0]
        time = concat_lps_sinc.shape[2]        
        re_cnn = cnn_out.view((batch, time, 512))
        
        concat_feat = torch.cat((re_cnn,ssl_feat_red), axis=1)
        out_lstm, (h, c) = self.mean_net_rnn(concat_feat)
        out_dense = self.mean_net_dnn(out_lstm) # (batch, seq, 1)       
        
        quality_att, _ = self.att_output_layer_quality (out_dense, out_dense, out_dense) 
        frame_quality = self.output_layer_quality(quality_att)
        frame_quality = self.sigmoid_(frame_quality)   
        quality_utt = self.qualaverage_score(frame_quality.permute(0,2,1))

        int_att, _ = self.att_output_layer_intell (out_dense, out_dense, out_dense) 
        frame_int = self.output_layer_intell(int_att)
        frame_int = self.sigmoid_(frame_int)   
        int_utt = self.intellaverage_score(frame_int.permute(0,2,1))

                
        return quality_utt.squeeze(1), int_utt.squeeze(1), frame_quality.squeeze(2), frame_int.squeeze(2)


#Adapt for the mosa net 
''' 
        self.att_output_layer_intell = nn.MultiheadAttention(128, num_heads=8)           
        self.output_layer_intell = nn.Linear(128, 1)
'''
def freeze_layers(model, layers_to_freeze):
    for name, layer in model.named_children():
        if name in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
'''

layers_to_freeze = ['att_output_layer_intell', 'output_layer_intell', 'intellaverage_score']
freeze_layers(model, layers_to_freeze)


classe dataset onde entra como os argumentos da lista:

mos_list com arquivos de áudio[0] e mos [1] .csv
features_data com arquivos dos caminhos para as features dos audios.pt .csv
'''
class MyDataset(Dataset):
    def __init__(self, mos_list, features_data) :
        self.mos_data = pd.read_csv(mos_list , header=None) #coluna 0 caminho do audio, coluna 1 score
        self.features_data = pd.read_csv(features_data, header=None) #coluna 0 caminho das features
       
    def __len__(self):
        return len(self.mos_data) 

        
    def __getitem__(self, idx):             
        
        wavfile = self.mos_data.iloc[idx, 0] #caminho do audio                    
        mos_score = float(self.mos_data.iloc[idx, 1]) #score do audio

        wav,_ = torchaudio.load(wavfile) #carrega o audio

        # Calculando STFT e LPS
        stft = torch.stft(wav[0], n_fft=512, hop_length=256, win_length=512, return_complex=False)
        magnitude = torch.abs(stft)
        lps = magnitude.transpose(0, 1).unsqueeze(0)
        
        feature_path = self.features_data.iloc[idx, 0] #caminho das features
        whisper_features = torch.load(feature_path) #carrega as features

        mos_tensor = torch.tensor(mos_score)
        
        return wav , lps , whisper_features, mos_tensor


def train(model, train_loader, optimizer, criterion, device, epochs, ckpt_path):
    # Definir o diretório de checkpoints
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Carregando checkpoint de {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
    else:
        print("Iniciando treinamento sem checkpoint.")
    
    model = model.to(device)
    model.train()  # Configura o modelo para o modo de treinamento

    patience = 5
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        print(f"=== Epoch {epoch+1}/{epochs} ===")
        running_loss = 0.0
        steps = 0

        for data in tqdm(train_loader):
            inputs, lps, whisper, labels_quality, labels_intell = data
            inputs, lps = inputs.to(device), lps.to(device)
            labels_quality, labels_intell = labels_quality.to(device), labels_intell.to(device)

            optimizer.zero_grad()
            output_quality, output_intell, frame_quality, frame_intell = model(inputs, lps, whisper)

            # Calcular as perdas
            label_frame_quality = frame_score(labels_quality, frame_quality)
            label_frame_intell = frame_score(labels_intell, frame_intell)
            loss_frame_quality = criterion(frame_quality, label_frame_quality)
            loss_frame_intell = criterion(frame_intell, label_frame_intell)
            loss_quality = criterion(output_quality.squeeze(1), labels_quality)
            loss_intell = criterion(output_intell.squeeze(1), labels_intell)

            # Somar todas as perdas para o gradiente
            loss = loss_quality + loss_frame_quality + loss_intell + loss_frame_intell

            # Backpropagation
            loss.backward()
            optimizer.step()
            steps += 1
            running_loss += loss.item()

        avg_train_loss = running_loss / steps
        print(f"Epoch {epoch+1} - Loss: {avg_train_loss:.4f}")

        # Salvar o modelo se a validação melhorar (a ser adicionada)
        if avg_train_loss < best_val_loss:
            print(f"Novo melhor modelo com perda {avg_train_loss:.4f}, salvando...")
            torch.save(model.state_dict(), ckpt_path)
            best_val_loss = avg_train_loss
            patience = 5  # Resetar paciência
        else:
            patience -= 1

        # Parar o treinamento se a paciência esgotar
        if patience == 0:
            print("Treinamento interrompido por paciência esgotada.")
            break
#Collate function TODO

    
def denorm(input_x):
    input_x = input_x*(5-0) + 0
    return input_x
    
def frame_score(y_true, y_predict):
    B,T = y_predict.size()  
    y_true_repeat = y_true.unsqueeze(1).repeat(1,T) #(B,T)  
    return y_true_repeat


#def teste( Função de teste baseada no código original)
def teste(ckpt_path , model, test_loader, device):
    checkpoint = torch.load(ckpt_path , map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MosPredictor().to(device)
    model.eval()

    total = 0
    correct = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss() #Mudar para MSE

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate final loss and accuracy
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='/MOSA-Net_Plus_Torch/Checkpoint', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    #define dataset 
    train_df = MyDataset('Trainer.csv' , 'TrainerWhisper.csv')
    train_loader = DataLoader(train_df, batch_size=1) #collate todo , num_workers todo

    test_df = MyDataset('Test.csv' , 'TestWhisper.csv')
    test_loader = DataLoader(test_df, batch=1) #collate todo , num_workers todo

    model = MosPredictor()

    #define optimizer loss and other stuffs  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    
    #put the train and the test loop here
    #Bring the .pth files in here 
        
    print("\n\n\n ######## Iniciando o fine tune #########   \n\n\n")
    
    '''
    Comentários para orientar 

    Iniciar o treinamento, função de teste e por último salvar o modelo 
    '''
    train(model, train_loader, optimizer, loss, device, epochs, args.outdir + '/model.pth')

    print("\n\n\n ######## Fine tune finalizado #########   \n\n\n")

    teste(args.outdir + '/model.pth', model, test_loader, device)
    
    torch.save(model.state_dict(), args.outdir + '/model.pth')
    
    
if __name__ == '__main__':
    main()

    
