import torch
import torch.nn as nn
import torch.nn.functional as F


class indrectAttention(nn.Module):
    def __init__(self, head=8, dim=192):
        super(indrectAttention,self).__init__()
        self.head = head
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(dim, dim * head)
        self.scale = torch.sqrt(torch.FloatTensor([dim])).cuda()

    def forward(self, drug, protein):
        bsz, d_ef,d_il = drug.shape
        bsz, p_ef, p_il = protein.shape
        
        drug_att = self.relu(self.linear(drug.permute(0, 2, 1))).view(bsz,self.head,d_il,d_ef)
        protein_att = self.relu(self.linear(protein.permute(0, 2, 1))).view(bsz,self.head,p_il,p_ef)
        interaction_map = torch.mean(self.tanh(torch.matmul(drug_att, protein_att.permute(0, 1, 3, 2)) / self.scale),1)
        
        Compound_atte = self.tanh(torch.sum(interaction_map, 2)).unsqueeze(1)
        Protein_atte = self.tanh(torch.sum(interaction_map, 1)).unsqueeze(1)
        
        drug_new = drug * Compound_atte
        protein_new = protein * Protein_atte
        return drug_new, protein_new
        

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

    
class SMFFDTA(nn.Module):
    
    def se_block(self, channels):
        se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  
            View((-1, channels)),     
            nn.Linear(channels, channels // 16, bias=False), 
            nn.ReLU(),              
            nn.Linear(channels // 16, channels, bias=False), 
            nn.Sigmoid(),            
            View((-1, channels, 1))   
        )
        return se_block
    
    
    def __init__(self, protein_MAX_LENGH = 1200, protein_kernel = [4,8,12],
                 drug_MAX_LENGH = 100, drug_kernel = [4,6,8],
                 conv = 32, char_dim = 128, dropout_rate = 0.1):
        super(SMFFDTA, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.dropout_rate = dropout_rate
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = protein_kernel
        
        self.drug_embed = nn.Embedding(65, 128, padding_idx=0)
        self.atom_embed = nn.Linear(38, 64)
        
        self.protein_embed = nn.Embedding(26, 64, padding_idx=0)
        self.sse_embed = nn.Embedding(3, 6, padding_idx=0)
        self.phyche_embed = nn.Embedding(7, 15, padding_idx=0)
        
        ## Squeeze-Excitation
        self.SE_drug = self.se_block(128)
        self.SE_atom = self.se_block(64)
        self.SE_protein = self.se_block(85)
        self.SE1 = self.se_block(64)
        self.SE2 = self.se_block(192)
        
        
        ## smi-64
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )  
        self.Drug_gru = nn.GRU(input_size=512, hidden_size=64, batch_first=True, bidirectional=True)
        self.Drug_adaptPool = nn.AdaptiveAvgPool1d(64)
        ## fp-64
        self.fp_linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        ## atomFeature-64
        self.Atom_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )  
        self.Atom_gru = nn.GRU(input_size=256, hidden_size=64, batch_first=True, bidirectional=True)
        self.Atom_adaptPool = nn.AdaptiveAvgPool1d(64)


        ## seq+sse+phyche-192
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=85, out_channels=256, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_gru = nn.GRU(input_size=512, hidden_size=192, batch_first=True, bidirectional=True)
        self.Protein_adaptPool = nn.AdaptiveAvgPool1d(192)
        
        self.Drug_max_pool = nn.MaxPool1d(85)
        self.Protein_max_pool = nn.MaxPool1d(589)
        
        self.maxPool = nn.MaxPool1d(2)
        
        self.attention = indrectAttention(head = 4, dim = 192)
        self.crossAtt = nn.MultiheadAttention(embed_dim=192, num_heads=8, batch_first=True)
        
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(384, 1024)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        torch.nn.init.constant_(self.out.bias, 5)
        

    def forward(self, drugSMI, drugFP, drugAtom, proteinSeq, proteinSSE, proteinPHYCHE):


        drugEmbed = self.drug_embed(drugSMI)            
        atomEmbed = self.atom_embed(drugAtom)               
        drugEmbed = drugEmbed.permute(0, 2, 1)
        drugfp = drugFP.to(torch.float)
        atomEmbed = atomEmbed.permute(0, 2, 1)
        
        proteinEmbed = self.protein_embed(proteinSeq)     
        sseEmbed = self.sse_embed(proteinSSE)               
        phycheEmbed = self.phyche_embed(proteinPHYCHE)     
        proteinEmbed = proteinEmbed.permute(0, 2, 1)
        sseEmbed = sseEmbed.permute(0, 2, 1)
        phycheEmbed = phycheEmbed.permute(0, 2, 1)
        

        ## SE-Block
        drugSE = self.SE_drug(drugEmbed)
        drugCnn = drugSE*drugEmbed.expand_as(drugEmbed)
        drugCnn = drugCnn + drugEmbed                      
        
        atomSE = self.SE_atom(atomEmbed)
        atomCnn = atomSE*atomEmbed.expand_as(atomEmbed)
        atomCnn = atomCnn + atomEmbed                      
        # atomCnn = atomCnn.repeat(1, 1, 2)
        
        proteinAll = torch.concat([proteinEmbed, sseEmbed, phycheEmbed], dim=1)      
        proteinSE = self.SE_protein(proteinAll)
        proteinCnn = proteinSE*proteinAll.expand_as(proteinAll)
        proteinCnn = proteinCnn + proteinAll             

                
        ## Drug-CNN 
        drugConv = self.Drug_CNNs(drugCnn)                 
        drugConv = drugConv.permute(0, 2, 1)
        drugGRU, _ = self.Drug_gru(drugConv)
        drugGRU = self.Drug_adaptPool(drugGRU)
        drugGRU = drugGRU.permute(0, 2, 1)               
        drugGRUSE = self.SE1(drugGRU)
        drug1 = drugGRUSE*drugGRU.expand_as(drugGRU)
        drug1 = drug1 + drugGRU
        ## Fingerprint-FC
        drugfp = self.fp_linear(drugfp)
        drugfp = drugfp.unsqueeze(2).repeat(1, 1, 85)      
        drugfpSE = self.SE1(drugfp)
        drug2 = drugfpSE*drugfp.expand_as(drugfp)
        drug2 = drug2 + drugfp
        ## Atom-CNN
        atomConv = self.Atom_CNNs(atomCnn)              
        atomConv = atomConv.permute(0, 2, 1)
        atomGRU, _ = self.Atom_gru(atomConv)
        atomGRU = self.Atom_adaptPool(atomGRU)
        atomGRU = atomGRU.permute(0, 2, 1)                
        atomGRUSE = self.SE1(atomGRU)
        drug3 = atomGRUSE*atomGRU.expand_as(atomGRU)
        drug3 = drug3 + atomGRU

    
        ## Protein-CNN  
        proteinConv = self.Protein_CNNs(proteinCnn)                 
        proteinConv = proteinConv.permute(0, 2, 1)
        proteinGRU, _ = self.Protein_gru(proteinConv)
        proteinGRU = self.Protein_adaptPool(proteinGRU)
        proteinGRU = proteinGRU.permute(0, 2, 1)          
        proteinGRUSE = self.SE2(proteinGRU)
        protein = proteinGRUSE*proteinGRU.expand_as(proteinGRU)
        protein = protein + proteinGRU

        
        ## feature fusion
        drugfuse = torch.cat([drug1, drug2, drug3], dim=1)      
        proteinfuse = self.maxPool(protein)                      

        drugSE2 = self.SE2(drugfuse)
        drug_mul = drugSE2*drugfuse.expand_as(drugfuse)
        drug_mul = drug_mul + drugfuse
        proteinSE2 = self.SE2(proteinfuse)
        protein_mul = proteinSE2*proteinfuse.expand_as(proteinfuse)
        protein_mul = protein_mul + proteinfuse       
              

        ## Interaction Block
        drugAtt, proteinAtt = self.attention(drugfuse, proteinfuse)
        drugAttSE = self.SE2(drugAtt)
        drugAtt_mul = drugAttSE*drugAtt.expand_as(drugAtt)
        drugAtt_mul = drugAtt_mul + drugAtt
        proteinAttSE = self.SE2(proteinAtt)
        proteinAtt_mul = proteinAttSE*proteinAtt.expand_as(proteinAtt)
        proteinAtt_mul = proteinAtt_mul + proteinAtt

        ## cross-attention
        drug_cross = drug_mul.permute(0, 2, 1)
        protein_cross = protein_mul.permute(0, 2, 1)
        drug_temp = drug_cross
        drug_cross, _ = self.crossAtt(drug_cross, protein_cross, protein_cross)
        protein_cross, _ = self.crossAtt(protein_cross, drug_temp, drug_temp)
        drug_cross = drug_cross.permute(0, 2, 1)
        protein_cross = protein_cross.permute(0, 2, 1)
        
        
        ## weighted Add  
        drugScore = self.softmax(drugAtt_mul * drug_cross)
        proteinScore = self.softmax(proteinAtt_mul * protein_cross)
        drugcat = drugScore * drugAtt_mul + (1-drugScore) * drug_cross + drugAtt_mul + drug_cross + drug_mul         
        proteincat = proteinScore * proteinAtt_mul +(1-proteinScore) * protein_cross + proteinAtt_mul + protein_cross + protein_mul

        
        ## Feature Concat
        drugcat = self.Drug_max_pool(drugcat).squeeze(2)
        proteincat = self.Protein_max_pool(proteincat).squeeze(2)
        
        pair = torch.cat([drugcat,proteincat], dim=1)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout1(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout2(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
    
        return predict
