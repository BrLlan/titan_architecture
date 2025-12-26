import torch
import torch.nn as nn
import math

#for each item we calcualte mean and variance and subteact mean
#and dvidie by absolute variance (sqauroot of square)
class LayerNormalization(nn.Module):
    
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        #tuning variables, learned by model, to ampliy values if 0-1 too restrtictive
        #nn.parameter makes it learnable
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

#feedforward has jsut two layers, 512 to 2048 and 2048 to 512
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        #bias is true by default
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        #input tensot -> matrix 1 to new tensor -> matrix 2 to output, droput in between
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class InputEmbeddings(nn.Module):

    #dmodel: output vector size
    #vocab_size: how many words inv ocubalry
    #nn.Embedding: A simple lookup table that stores embeddings of a fixed dictionary and size.
    #This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
#Encodings are added to emebdded vectors, 
#simply cocntenated, to encode positon in input
#d_model is size of psoitonal encoidng,seq_len: max length of sentence, 
#dropout: to make it less overfit (why?)
    #size: seq_len X dmodel
    #fo for each token/word a full embedding vector
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        #word inside sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        #buffer saves state of model
        self.register_buffer('pe', pe)

    def forward(self, x):
        #requires_grad false: vector is not learned, fixed
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            #we need norm because normlaisation happens in the other layer,
            #which is skipped here
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            #in pper, order of norm and subalyer was reverse here
            return x + self.dropout(sublayer(self.norm(x)))


#input matrix covnerted to three matrices, QKV, same shape as input
        #time three w matrixed, results in three matrices
        #seqXd_model X d_modelxd_model -> seq, d_model
        #we then split those into h molar matrices, along embedding dminesions
        #each is seq long
        #h = number of attneiton heads
        #each head will have access to full sentence, but different part of embedding of each word
        #we apply the attention to each ofthese smaller matrices using this formula which will give us smaller matrices as a result 
        #then we combine them back so we can cut them back just like the paper says 
        #so concatenation of head 1 up to head Edge 
        #and finally we multiply it by w o to get the multi-head attention output 
        #which again is a Matrix 
        #Matrix that has the same Dimension as the input Matrix as you can see it's the output of the multihead attention is also seq by D_model
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo hX d_v,d_model (d_v = d_k)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    #callable from otuside without needing isntance of class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #print("attention_scores",attention_scores.shape)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    
    #mask: reudces attentions core between twow rods we want to not have attention
    #to small vlaue beofre softmax, thus way attention score becomes essentially zero
    #just applies -1e9 to fields, softma
    #eg for futrue words or for padding words to fill seq length
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        #we split mebeddings here?
        #h * d_k = d_model
        #we trasnposne for h to be second dimension
        #that way each head will see seq length x d_k
        #each head sees each aprt of sentence but only as aprt of mebedding?
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #at this point we got smaller attention amtrices, h of them. cocnatenate here
        # Combine all the heads together
        #contguous creates copy? memory usage? He says do it in palce, but docs say copy
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    

class EncoderBlock(nn.Module):
    #we call it self-attention because in the
    #case of the encoder it is applied to the same input with three different roles
    #the role of query of the key and the value
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    #src_mask: applied to input of encoder, fpr padding words
    #we don't want them itneractign with other words
    def forward(self, x, src_mask):
        #called self attention ebcause key, value and query are all x, the input
        #that way each word of sentenc eis itneractign with all toehr words
        #attention suually input and output, here onyl inputs
        #later: queries coming formd ecoder use keys and vlaues from encoder
        #called cross attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x), mask

#queries coming formd ecoder use keys and vlaues from encoder
#called cross attention
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    #src_mask from encoder, for input
    #tgt_mask,t arget maks, fromd ecoder, fort ergat
    #he says they are speerate because twio different langugaes
    #I must test if that won't work for chat/prompts
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
#lineaer layer after decoder
#"projects" the embeddings back to the vocabulary
#so 512 to 30.000?
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        #in the video there was softmax already here?
        #reason: https://github.com/hkproj/pytorch-transformer/issues/7
        return self.proj(x)
    
class Transformer(nn.Module):

    #src_embed, tgt_embed_ embeddingas for didferent languages. 
    #I suppsoe Ic an jsut inut the same one if I trian on english?
    #or maybe there is multilangugae embedding?
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer, number_p_tokens: int, d_model: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        #prefix tokens
        self.number_p_tokens = number_p_tokens
        self.src_prefix = nn.Parameter(torch.randn(self.number_p_tokens, d_model))

    #seperate fucntions becuase we don't need to recalcualte encoder each time
    #we also want output for visulaiton 
    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)

        batch_size = src.size(0)
        prefix = self.src_prefix.unsqueeze(0)      # (1, number_p_tokes, d_model)
        prefix = prefix.expand(batch_size, -1, -1) # (batch_size, number_p_tokes, d_model)
        src = torch.cat([prefix, src], dim=1)      # (batch_size, number_p_tokes + seq_len, d_model)

        src = self.src_pos(src) #With adapted seq_len to account for more tokens

        #we need to adapt the src_mask here to fit the new, expanded effective_seq_len
        if src_mask is not None:
            prefix_mask = torch.ones(
                (batch_size, 1, 1, self.number_p_tokens),
                device=src_mask.device,
                dtype=src_mask.dtype
            )
            src_mask = torch.cat([prefix_mask, src_mask], dim=-1)
            # now: (batch_size, 1, 1, effective_seq_len)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        tgt:       (batch_size, seq_len)
        tgt_mask:  (batch_size, 1, seq_len, seq_len)
        """

        #embed target tokens
        tgt = self.tgt_embed(tgt)     # (batch_size, seq_len, d_model)
        tgt = self.tgt_pos(tgt)

        batch_size, seq_len, _ = tgt.shape
        p = self.number_p_tokens
        device = tgt.device

        #prepend persistent tokens to decoder as well
        prefix = self.src_prefix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, number_p_tokens, d_model)
        tgt = torch.cat([prefix, tgt], dim=1)# (batch_size, new_seq_len, d_model)
        
        #adjust target mask
        if tgt_mask is not None:
            # tgt_mask is (batch_size, 1, seq_len, seq_len)
            new_seq_len = seq_len + p

            new_mask = torch.zeros(
                (batch_size, 1, new_seq_len, new_seq_len),
                device=device,
                dtype=tgt_mask.dtype
            )

            #prefix rows: prefix can attend to everything, no issue
            new_mask[:, :, :p, :] = 1

            #all token rows can attend to prefix, sicne they all use the 'memory'
            new_mask[:, :, p:, :p] = 1

            #adapt tgt_mask shape (greedy decoding fix)
            if tgt_mask.dim() == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            elif tgt_mask.dim() == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                tgt_mask = tgt_mask.unsqueeze(1)

            """new_mask[:, :, P:, P:] = tgt_mask"""

            #maks "step size": we need to adjsut mask size because we use inference
            #token after token
            t = tgt.size(1)

            #we must use the orignal target maks and not mask the newly added prefix tokens
            #new_mask[:, :, p:p+t, p:p+t]  "take all elements starting form the index of prefix tokens and end at prefix tokesn + current step"
            #so currently generated target tokens, after the prefix tokens
            #tgt_mask[:, :, -t:, -t:] "take the step size index from the end"
            #"take a bigger and bigger part of the tgt_mask (causal + padding mask) as we progress with inference"
            new_mask[:, :, p:p+t, p:p+t] = tgt_mask[:, :, -t:, -t:]

            tgt_mask = new_mask


        #sent to decoder
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
#he says we can use the same structure for every task, not jsut trasnaltion
#tgt seq length? How do we know that before?
#N decoder and encoder blocks. Always the same?
#def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, number_p_tokens: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len+number_p_tokens, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer,number_p_tokens,d_model)
    
    # Initialize the parameters
    #xavier uniform initliases apramters non-randomly to make trianing faster
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer