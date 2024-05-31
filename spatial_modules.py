"""
The implementation of spatial modules, self-attention (SA) and graph attention (GA).
It's a consice and simple use of channel-wise attention / graph.

! please install torch_geometric and insert the following code in nice_stand.py
"""


class channel_attention(nn.Module):
    def __init__(self, sequence_num=250, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64), 
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, 'b o c s->b o s c')
        temp_query = rearrange(self.query(temp), 'b o s c -> b o c s')
        temp_key = rearrange(self.key(temp), 'b o s c -> b o c s')

        channel_query = temp_query
        channel_key = temp_key

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.einsum('b o c s, b o m s -> b o c m', channel_query, channel_key) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum('b o c s, b o c m -> b o c s', x, channel_atten_score)

        out = rearrange(out, 'b o c s -> b o s c')
        out = self.projection(out)
        out = rearrange(out, 'b o s c -> b o c s')
        return out


from torch_geometric.nn import GATConv
class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        # self.conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, heads=1)

        self.num_channels = 64
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()

    def forward(self, x):

        batch_size, _, num_channels, num_features = x.size()
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, depth=3, n_classes=4, **kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            ),
            # ResidualAdd(
            #     nn.Sequential(
            #         nn.LayerNorm(250),
            #         channel_attention(),
            #         nn.Dropout(0.3),
            #     )
            # ),
            PatchEmbedding(emb_size),
            FlattenHead(emb_size, n_classes)
        )
