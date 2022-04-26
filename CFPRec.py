import torch
from torch import nn

import math

class TFPRecEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        # get vocab size for each feature
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num+1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num+1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num+1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num+1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num+1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb),1)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size 
        self.heads      = heads
        self.head_dim   = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values  = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys    = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out  = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query, dist_matrix):
        # N = query.shape[0] # number of samples, TODO: make it batch

        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(query)
        
        # Split the embedding into self.heads different pieces
        # Multi head
        # [len, embed_size] --> [len, heads, head_dim]
        values    = values.reshape(value_len, self.heads, self.head_dim)
        keys      = keys.reshape(key_len, self.heads, self.head_dim)
        queries   = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])  # [heads, query_len, key_len]

        # add dist_matrix in weight
        if dist_matrix is not None:
            dist_matrix[dist_matrix==0] = math.inf
            dist_matrix += 1
            dist_matrix = 1 / dist_matrix
            
            energy += dist_matrix

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2) #[heads, query_len, key_len]

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        ) # [query_len, key_len]

        out = self.fc_out(out) # [query_len, key_len]

        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1     = nn.LayerNorm(self.embed_size)
        self.norm2     = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, dist_matrix):
        attention = self.attention(value, key, query, dist_matrix) #[len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x       = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out     = self.dropout(self.norm2(forward + x))
        return out

class TFPRecEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
        ):

        super(TFPRecEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq, dist_matrix):
        
        embedding = self.embedding_layer(feature_seq) #[len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out, dist_matrix)

        return out

class Attention(nn.Module):
    def __init__(
        self,
        qdim,
        kdim,
    ):
        super().__init__()

        self.expansion = nn.Linear(qdim, kdim)


    def forward(self, query, key, value):
        q = self.expansion(query) # [embed_size]
        weight = torch.softmax(torch.inner(q, key), dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        out = torch.sum(torch.mul(value, weight), 0) # sum([len, embed_size] * [len, 1])  -> [embed_size]
        
        return out

class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size, dropout_p):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.poi_linear = nn.Linear(input_size, vocab_size["POI"])
        self.cat_linear = nn.Linear(input_size, vocab_size["cat"])
        self.hour_linear = nn.Linear(input_size, vocab_size["hour"])

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, masked_output, masked_target):
        """calculate 

        Args:
            masked_output ([type]): [description]
            masked_target ([type]): [description]

        Returns:
            [type]: [description]
        """
        aux_loss = 0.
        poi_out = self.poi_linear(self.dropout(masked_output))
        poi_loss = self.loss_func(poi_out, masked_target[0])

        cat_out = self.cat_linear(self.dropout(masked_output))
        cat_loss = self.loss_func(cat_out, masked_target[1])

        hour_out = self.hour_linear(self.dropout(masked_output))
        hour_loss = self.loss_func(hour_out, masked_target[2])

        aux_loss = poi_loss + cat_loss + hour_loss
        return aux_loss

class TFPRec(nn.Module):
    def __init__(
        self,
        vocab_size,
        f_embed_size = 2,
        num_encoder_layers = 1,
        num_lstm_layers = 1,
        num_heads = 1,
        forward_expansion = 2,
        dropout_p = 0.1,
        back_step = 2,
        aux_train = False,
        mask_prop = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5
        self.backstep = back_step
        self.aux_train = aux_train
        self.mask_prop = mask_prop

        # LAYERS
        self.embedding = TFPRecEmbedding(
            f_embed_size, 
            vocab_size
        )
        self.encoder = TFPRecEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.aux_loss = MaskedLM(
            input_size=self.total_embed_size, 
            vocab_size = vocab_size,
            dropout_p=dropout_p
        )
        self.lstm = nn.LSTM(
            input_size = self.total_embed_size,
            hidden_size = self.total_embed_size,
            num_layers = num_lstm_layers,
            dropout = 0
        )
        self.intra_seq_attention = Attention(
            qdim = f_embed_size, 
            kdim = self.total_embed_size
        )

        self.inter_seq_attention = Attention(
            qdim = f_embed_size, 
            kdim = self.total_embed_size
        )

        self.final_attention = Attention(
            qdim = f_embed_size, 
            kdim =self.total_embed_size
        )

        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size*forward_expansion), nn.LeakyReLU(),
                                        nn.Dropout(dropout_p), nn.Linear(self.total_embed_size*forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()


    def gen_random_mask(self, input, mask_prop):
        """generate mask (index) for long term sequence for auxiliary training

        Args:
            input ([[long term sequence]]): long term sequences in a sample

        Returns:
            index_list (): 
            sample_list ():
            target_list (): 
        """
        index_list, sample_list = [], []
        poi_target_list, cat_target_list, user_target_list, hour_target_list, day_target_list = [], [], [], [], []
        for seq in input: # each long term sequences
            feature_seq, dist_matrix = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int() 
            masked_index = torch.randperm(seq_len-1) + torch.tensor(1)
            masked_index = masked_index[:mask_count] # randomly generate mask index
            index_list.append(masked_index)

            # record masked true values
            poi_target_list.append(feature_seq[0, masked_index])
            cat_target_list.append(feature_seq[1, masked_index])
            hour_target_list.append(feature_seq[3, masked_index])
            
            # mask long term sequences
            feature_seq[0, masked_index] = self.vocab_size["POI"] # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"] # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"] # mask hour

            dist_matrix[:,masked_index] = 0.  # mask dist_matrix

            sample_list.append((feature_seq, dist_matrix))

        target_list = (
            torch.cat(poi_target_list, dim=0),
            torch.cat(cat_target_list, dim=0), 
            torch.cat(hour_target_list, dim=0), 
        )

        return index_list, sample_list, target_list
    
    def forward(self, sample):
        # process input sample
        long_term_seqences = sample[:-1] #[(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq]),((dist_matrix))],[(seq2)],...]
        short_term_sequence = sample[-1]
        short_term_pre = short_term_sequence[0][:, :-self.backstep-1]
        short_term_after_time = short_term_sequence[0][3, -self.backstep:]
        user_id = short_term_sequence[0][2, 0]
        target = short_term_sequence[0][0, -self.backstep-1]

        # mask input sequences
        mask_index, masked_seqences, masked_targets = None, None, None
        if self.aux_train:
            mask_index, masked_seqences, masked_targets = self.gen_random_mask(long_term_seqences, self.mask_prop)
            long_term_seqences = masked_seqences

        # long term section
        long_term_out = [] #[6*10, 8*10, ...]
        for seq in long_term_seqences:
            output = self.encoder(feature_seq=seq[0], dist_matrix=seq[1]) # [len, emb_size]
            long_term_out.append(output) # [seq_num, len, emb_size]
        
        # calculate auxiliary loss
        aux_loss = 0.
        if self.aux_train:
            aux_hidden = []
            for seq_index, seq_mask_id in enumerate(mask_index):
                aux_hidden.append(long_term_out[seq_index][seq_mask_id])
            aux_hidden = torch.squeeze(torch.cat(aux_hidden, dim=0)) # [masked_visit_num, embed_size]
            # masked_targets = torch.cat(masked_targets, dim=0) # [masked_visit_num, 1]
            aux_loss = self.aux_loss(aux_hidden, masked_targets)
        
        # short term section

        ## previous states
        embedding = torch.unsqueeze(self.embedding(short_term_pre), 0) # [batch(1), len, emb_size]
        output, _ = self.lstm(embedding) 
        pre_state = torch.squeeze(output)

        # later states
        inter_seq_reps = []
        user_embed = self.embedding.user_embed(user_id)
        # get representations for each future time stamp
        for time in short_term_after_time:
            time_embed = self.embedding.hour_embed(time) # [f_embed_size]
            intra_seq_reps = []
            # get representations for each sequence for the time
            for H in long_term_out: # attention: long term + time
                S = self.intra_seq_attention(time_embed, H, H) # [embed_size]
                intra_seq_reps.append(S)
            intra_seq_reps = torch.stack(intra_seq_reps) # [num_seq, embed_size]
            h_t = self.inter_seq_attention(user_embed, intra_seq_reps, intra_seq_reps)
            inter_seq_reps.append(h_t)
        inter_seq_reps = torch.stack(inter_seq_reps) #[time_num, embed_size]

        # final output

        ## concat pre and after state
        h = torch.cat((pre_state, inter_seq_reps))
        final_att = self.final_attention(user_embed, h, h)
        output = self.out_linear(final_att)

        label = torch.unsqueeze(target,0)
        pred = torch.unsqueeze(output,0)

        loss = self.loss_func(pred, label)
        final_loss = loss + aux_loss
        # final_loss = loss

        return final_loss, output
    
    def predict(self, sample):
        _, pred_raw = self.forward(sample)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0,-self.backstep-1]

        return ranking, target

if __name__ == "__main__":
    pass
