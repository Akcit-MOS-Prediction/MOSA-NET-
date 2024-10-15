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
def freeze( model):
    for param in model.parameters():
        param.requires_grad = False


'''
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

        


#Collate function TODO

    
def denorm(input_x):
    input_x = input_x*(5-0) + 0
    return input_x
    
def frame_score(y_true, y_predict):
    B,T = y_predict.size()  
    y_true_repeat = y_true.unsqueeze(1).repeat(1,T) #(B,T)  
    return y_true_repeat


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

    #define optimizer loss and other stuffs  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    
    
    #put the train and the test loop here
        
    print("\n\n\n ######## Iniciando o fine tune #########   \n\n\n")
    
    #training loop
    #Testing 

    
    #torch.save(model.state_dict(), args.outdir + '/model.pth')
    
    
if __name__ == '__main__':
    main()

    
