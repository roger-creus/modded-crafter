import os
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
import numpy as np

class SeqVAE(nn.Module):
    def __init__(self,frames,f_dim,z_dim,conv_dim,hidden_dim):
        super(SeqVAE,self).__init__()
        self.f_dim = f_dim
        self.z_dim = z_dim
        
        # seq len
        self.frames = frames
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim

        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1, bidirectional=True,batch_first=True)
        self.f_mean = nn.Linear(self.hidden_dim*2, self.f_dim)
        self.f_logvar = nn.Linear(self.hidden_dim*2, self.f_dim)

        self.f_mean_drop = nn.Dropout(0.5)
        self.f_logvar_drop = nn.Dropout(0.5)

        self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True,batch_first=True)
        self.z_rnn = nn.RNN(self.hidden_dim*2, self.hidden_dim,batch_first=True) 
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        
        self.z_mean_drop = nn.Dropout(0.5)
        self.z_logvar_drop = nn.Dropout(0.5)

        #### Visual Encoder #### 
        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout2d(0.5),
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_fc = nn.Linear(hidden_dims[-1] * 2 * 2, self.conv_dim) 
        self.bnf = nn.BatchNorm1d(self.conv_dim) 
        self.drop_fc = nn.Dropout(0.5)
        ################################

        #### Visual Decoder #### 
        modules = []

        self.decoder_fc = nn.Linear(self.conv_dim, hidden_dims[-1] * 2 * 2)
        self.decoder_bn = nn.BatchNorm1d(2 * 2 * hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.1)),
                    nn.Dropout(0.5),
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.LeakyReLU(0.1),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                nn.Tanh()
                            )

        ################################
        
        #### Initialization ####
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,1)
            elif isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu') 
        
        #nn.init.xavier_normal_(self.final_layer.weight, nn.init.calculate_gain('tanh'))
        ################################
    
    def encode_frames(self,x):
        x = x.view(-1,3,64,64)  # Batchwise stack the 8 images for applying convolutions parallely
        
        x = self.encoder(x)
        x = x.view(-1,2*2*512)  # Flattent the encoder CNN outputs
        x = self.encoder_fc(x)
        
        x = x.view(-1, self.frames, self.conv_dim) # Reshape to (, seq_len, latent_dim)
        return x

    def decode_frames(self, zf):
        x = zf.view(-1,self.f_dim+self.z_dim) # For batchnorm1D to work, the frames should be stacked batchwise
        x = self.decoder_fc(x)
        x = x.view(-1,512,2,2)  # The 8 frames are stacked batchwise
        x = self.decoder(x)
        x = self.final_layer(x)
        return x.view(-1, self.frames ,3, 64, 64) # Output is of shape (batch_size, seq_len, channels, width, height)

    def reparameterize(self,mean,logvar):
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out,_ = self.f_lstm(x)
        mean = self.f_mean(self.f_mean_drop(lstm_out[:,self.frames-1])) # The forward and the reverse are already concatenated
        logvar = self.f_logvar(self.f_logvar_drop(lstm_out[:,self.frames-1])) # TODO: Check if its the correct forward and reverse
        return mean, logvar, self.reparameterize(mean,logvar)
    
    def encode_z(self,x,f):
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        lstm_out,_ = self.z_lstm(torch.cat((x, f_expand), dim=2))
        rnn_out,_ = self.z_rnn(lstm_out)
        mean = self.z_mean(self.z_mean_drop(rnn_out))
        logvar = self.z_logvar(self.z_logvar_drop(rnn_out))
        return mean,logvar,self.reparameterize(mean,logvar)

    def forward(self, x):
        conv_x = self.encode_frames(x)
        f_mean,f_logvar,f = self.encode_f(conv_x)
        z_mean,z_logvar,z = self.encode_z(conv_x,f)
        f_expand = f.unsqueeze(1).expand(-1,self.frames,self.f_dim)
        zf = torch.cat((z,f_expand),dim=2)
        recon_x = self.decode_frames(zf)
        return f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x

def loss_fn(original_seq,recon_seq,f_mean,f_logvar,z_mean,z_logvar):
    mse = F.mse_loss(recon_seq,original_seq,reduction='sum')
    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    kld_z = -0.5 * torch.sum(1 + z_logvar - torch.pow(z_mean,2) - torch.exp(z_logvar))
    return mse + kld_f + kld_z
  
class Trainer(object):
    def __init__(self,model,device,train,test,trainloader,testloader,epochs,batch_size,learning_rate,nsamples,sample_path,recon_path,checkpoints):
        self.trainloader = trainloader
        self.train = train
        self.test = test
        self.testloader = testloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(),self.learning_rate)
        self.samples = nsamples
        self.sample_path = sample_path
        self.recon_path = recon_path
        self.test_f = torch.randn(self.samples,self.model.f_dim,device=self.device)
        self.test_z = torch.randn(self.samples,model.frames,model.z_dim,device=self.device)
        f_expand = self.test_f.unsqueeze(1).expand(-1,model.frames,model.f_dim)
        self.test_zf = torch.cat((self.test_z,f_expand),dim=2)
        self.epoch_losses = []

        self.image1 = torch.load('image1.sprite')
        self.image2 = torch.load('image2.sprite')
        self.image1 = self.image1.to(device)
        self.image2 = self.image2.to(device)
        self.image1 = torch.unsqueeze(self.image1,0)
        self.image2= torch.unsqueeze(self.image2,0)
    
    def save_checkpoint(self,epoch):
        torch.save({
            'epoch' : epoch+1,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'losses' : self.epoch_losses},
            self.checkpoints)
        
    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_frames(self,epoch):
        with torch.no_grad():
           recon_x = self.model.decode_frames(self.test_zf) 
           recon_x = recon_x.view(16,3,64,64)
           torchvision.utils.save_image(recon_x,'./Full/%s/epoch%d.png' % (self.sample_path,epoch))
    
    def recon_frame(self,epoch,original):
        with torch.no_grad():
            _,_,_,_,_,_,recon = self.model(original) 
            image = torch.cat((original,recon),dim=0)
            print(image.shape)
            image = image.view(16,3,64,64)
            torchvision.utils.save_image(image,'./Full/%s/epoch%d.png' % (self.recon_path,epoch))

    def style_transfer(self,epoch):
        with torch.no_grad():
            conv1 = self.model.encode_frames(self.image1)
            conv2 = self.model.encode_frames(self.image2)
            _,_,image1_f = self.model.encode_f(conv1)
            image1_f_expand = image1_f.unsqueeze(1).expand(-1,self.model.frames,self.model.f_dim)
            _,_,image1_z = self.model.encode_z(conv1,image1_f)
            _,_,image2_f = self.model.encode_f(conv2)
            image2_f_expand = image2_f.unsqueeze(1).expand(-1,self.model.frames,self.model.f_dim)
            _,_,image2_z = self.model.encode_z(conv2,image2_f)
            image1swap_zf = torch.cat((image2_z,image1_f_expand),dim=2)
            image1_body_image2_motion = self.model.decode_frames(image1swap_zf)
            image1_body_image2_motion = torch.squeeze(image1_body_image2_motion,0)
            image2swap_zf = torch.cat((image1_z,image2_f_expand),dim=2)
            image2_body_image1_motion = self.model.decode_frames(image2swap_zf)
            image2_body_image1_motion = torch.squeeze(image2_body_image1_motion,0)
            os.makedirs(os.path.dirname('./Full/transfer/epoch%d/image1_body_image2_motion.png' % epoch),exist_ok=True)
            torchvision.utils.save_image(image1_body_image2_motion,'./Full/transfer/epoch%d/image1_body_image2_motion.png' % epoch)
            torchvision.utils.save_image(image2_body_image1_motion,'./Full/transfer/epoch%d/image2_body_image1_motion.png' % epoch)



    def train_model(self):
       self.model.train()
       for epoch in range(self.start_epoch,self.epochs):
           losses = []
           print("Running Epoch : {}".format(epoch+1))
           for i,data in enumerate(self.trainloader,1):
               data = data.to(device)
               self.optimizer.zero_grad()
               f_mean,f_logvar,f,z_mean,z_logvar,z,recon_x = self.model(data)
               loss = loss_fn(data,recon_x,f_mean,f_logvar,z_mean,z_logvar)
               loss.backward()
               self.optimizer.step()
               losses.append(loss.item())
           meanloss = np.mean(losses)
           self.epoch_losses.append(meanloss)
           print("Epoch {} : Average Loss: {}".format(epoch+1,meanloss))
           self.save_checkpoint(epoch) 
           self.model.eval()
           self.sample_frames(epoch+1)
           sample = self.test[int(torch.randint(0,len(self.test),(1,)).item())]
           sample = torch.unsqueeze(sample,0)
           sample = sample.to(self.device)
           self.recon_frame(epoch+1,sample)
           self.style_transfer(epoch+1)
           self.model.train()
       print("Training is complete")

if __name__ == '__main__':
    vae = FullQDisentangledVAE(frames=8,f_dim=64,z_dim=32,hidden_dim=512,conv_dim=1024) 
    sprites_train = Sprites('./indexed-sprites/lpc-dataset/train/', 6687)
    sprites_test = Sprites('./indexed-sprites/lpc-dataset/test/',873)
    trainloader = torch.utils.data.DataLoader(sprites_train,batch_size=64,shuffle=True,num_workers=4) 
    testloader = torch.utils.data.DataLoader(sprites_test,batch_size=1,shuffle=True,num_workers=4)
    device = torch.device('cuda:0')
    trainer = Trainer(vae,device,sprites_train,sprites_test,trainloader,testloader,epochs=100,batch_size=64,learning_rate=0.0002,checkpoints='disentangled-vae.model',nsamples = 2,sample_path='samples',
            recon_path='recon') 
    trainer.load_checkpoint()
    trainer.train_model()