import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def apply_attention(self, Q, K, V, mask = None):
      
        attention_raw = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

                
        #attention_raw = [batch size, n heads, query len, key len]
                
        if mask is not None:
            attention_raw = attention_raw.masked_fill(mask == 0, -1e10)
        
       
        attention = torch.softmax(attention_raw, dim=-1)
      
                
        attention = self.dropout(attention)
        
        #attention = [batch size, n heads, query len, key len]
        
        x = torch.matmul(attention, V)
       
        
        # return x, attention
        # return x.cpu(), attention.cpu()
        # return x.detach().cpu().numpy(), attention.detach().cpu().numpy()
        # return x.detach().cpu().tolist(), attention.detach().cpu().tolist()
        return np.array(x.detach().cpu().tolist()), np.array(attention.detach().cpu().tolist())

    
    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
        
        x, attention = self.apply_attention(Q, K, V, mask)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
