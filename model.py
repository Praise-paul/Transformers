import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module): ## This embedds letters into vectors and in this code it is 512.
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model #Dimension of the model
        self.vocal_size = vocab_size # Size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Scale the embeddings by the square root of d_model
    
class PositionalEncoding(nn.Module): ## Adding positional encoding to the input embeddings
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout) ## This is to avoid overfitting

        #Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_length, d_model)

        # Create a vector of shape seq_lenght

        position = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Apply SINE finc() for even indices and COSINE for odd indices

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_length, d_model)

        self.register_buffer('pe', pe)  # Register the positional encoding as a buffer so it is not a parameter of the model

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1)) ## Multiplied by the input
        self.bias = nn.Parameter(torch.zeros(1)) ## Added to the input

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.alpha) + self.bias # Apply the layer normalization

class FeedForwardBlock(nn.Module): ## This is the feed forward network
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module): ## This is the multi head attention block
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) // math.sqrt(d_k) # (Batch, n_heads, seq_len, d_k) @ (Batch, n_heads, d_k, seq_len) -> (Batch, n_heads, seq_len, seq_len)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # Mask the attention scores
        attention_scores = attention_scores.softmax(dim = -1) # Apply softmax to the attention scores

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores@ value), attention_scores # (Batch, n_heads, seq_len, d_k) @ (Batch, n_heads, d_k, seq_len) -> (Batch, n_heads, seq_len, d_model)
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1, 2) # (Batch, seq_len, n_heads, d_k) -> (Batch, n_heads, seq_len, d_k)
        key = query.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1, 2) # (Batch, seq_len, n_heads, d_k) -> (Batch, n_heads, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1, 2) # (Batch, seq_len, n_heads, d_k) -> (Batch, n_heads, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # (Batch, n_heads, seq_len, d_k) -> (Batch, seq_len, d_model)

        # (Batch, n_heads, seq_len, d_k) -> (batch, seq_len, h, d_k, d_model) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1, self.d_k)

        return self.w_o(x) # (Batch, seq_len, d_model)-> (Batch, seq_len, d_model)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization() # Layer normalization is applied after the residual connection to avoid overfitting
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # Add the input to the output of the sublayer


class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = self.feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # Two residual connections, one for the self attention block and one for the feed forward block

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # Apply the self attention block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layer: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x

class Decoder(nn.Module):
    def __init__(self, layer: nn.ModuleList) -> None:
        super().__init__()
        self.layer = layer
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layer:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x): ## (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) # Apply softmax to the output of the projection layer

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, target_embed: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int , d_model: int = 512, N: int = 6, n_heads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    #Create the embeddings layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_vocab_size)

    #Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    #Create the encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    #Create the decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attentionblock = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        decoder_block = DecoderBlock(decoder_self_attentionblock, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    #Create the encoder and decoder

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #Create the projection Layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    #Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)
    

    # Initialize the parameters of the model

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer