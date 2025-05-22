import torch 
import torch.nn as nn
import math 

class InputEmbeddings(nn.Module):
    
    def __init__(self , d_model : int , vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model) #this will map each number to a unique vector(same for one number every time)
        #of size 512 , which is then learnt by our model    
        
    def forward(self , x):
        return self.embedding(x)*math.sqrt(self.d_model)#apparently it is written in the paper :(i have no clue which paper for now) that we multiply it by the root(d_model)
    #i have no idea what is happening here , oh wait i understand
    #we are basically feeding the forward function a number which is the input id of the word in our vocab and it is being worked upon by nn.embedding
        
class PositionalEncoding(nn.Module):
    def __init__(self , d_model : int , seq_len : int , dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) #dropout is to make the model less overfit , but why tho and how does it work ? 
        #it basically sets the value of some values to 0 and this makes the model adapt to lesser data making it less overfit
        
        #create a matrix of shape (seq_len , d_model )
        pe = torch.zeros(seq_len , d_model\ )
        
        #create a vector of shape seq_len,1
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # Shape: (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ) 
        #now we apply the sin and cosine position even and add
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[: , 1::2] = torch.cos(position * div_term)
        
        #something about how we will take in a batch of sentences and thus we need the model to be able to work on them , work on each sentence separately
        pe = pe.unsqueeze(0) # makes this (1, seq_len , d_model) , unsqueeze is a function which allows us to add a new dimension at the input position 
        
        self.register_buffer('pe' , pe) # when you have a tensor which you want to keep not as learnt parameter , but you want it saved we will save it as a buffer
        #ok so this will basically allow us to work with this matrix we have created PE , positional encoding , on other devices too and 
        #it is stored in .state_dict() , but then it is not treated as a learnable param and will nit be optimized 
        
    def forward(self , x):
        x = x + (self.pe[: , :x.shape[1] , :]).requires_grad_(False) #why do we need to add it in this way , why not straight away x + self.pe?
        #ok so we need to this as at some step maybe the seq_len of x is less than what we have entered , assume it to be max_len , the shenaningans we have
        #performed above allow us to slice the the Positional encoding to only include the positions up to the current seq_len , that is we have subtracted from 
        #max_len such that now it is equal to seq_len and thus self.pe[: , :x.shape[1] , :] produces a tensor of shape (1 , seq_len , d_model ) , where
        #seq_len is the max sequence length of x(the input batch ) which can now be Broadcasted across the batch dimension x and et voila 
        return self.dropout(x   )
        