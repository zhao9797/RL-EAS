import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNEncoder, self).__init__()
        self.cnn = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1)
        self.dropout = nn.Dropout()
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.cnn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.max(x, dim=2)[0]
        return x

class MetricSum(nn.Module):

    def __init__(self, opts):
        super(MetricSum, self).__init__()
        self.embedding = nn.Embedding(opts.vocab_size, opts.embedding_dim)
        self.sentence_encoder = CNNEncoder(opts.embedding_dim, 230)
        self.classifier = nn.Linear(230, 1)

    def forward(self, docs):
        batch_len = len(docs)
        # Check size of tensors
        logits = []
        for i in range(batch_len):
            s = self.embedding(docs[i])
            s = self.sentence_encoder(s)
            s = self.classifier(s).squeeze(1)
            s = s.sigmoid()
            logits.append(s)
        return logits
    def loss(self, x, y):
        loss = 0.0
        for i in range(len(x)):
            xx = x[i]
            yy = y[i]
            temp = torch.zeros((len(xx)), device=xx.device)
            temp[yy] = 1.0
            loss += F.binary_cross_entropy(xx, temp)
        return loss


# class HGAT(nn.Module):
#     def __init__(self, config):
#         super(HGAT, self).__init__()
#         self.config = config
#         hidden_size = config.hidden_size
#         self.embeding = nn.Embedding(config.class_nums, hidden_size)
#         self.relation = nn.Linear(hidden_size, hidden_size)
#         self.fc1 = nn.Linear(3*hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 1)
#         self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(config.gat_layers)])

#     def forward(self, x, pos1, pos2, mask=None, pretrian=False):
#         p = torch.arange(self.config.class_nums, device = x.device).long()
#         p = self.embeding(p)
#         p = self.relation(p)
#         p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))  # bcd
#         x, p = self.gat_layer(x, p, mask)
#         e1 = self.entity_trans(x, pos1)
#         e2 = self.entity_trans(x, pos2)

#         p = torch.cat([p, e1.unsqueeze(1).expand_as(p), e2.unsqueeze(1).expand_as(p)], 2)
#         p = self.fc1(p)
#         p = torch.tanh(p)
#         p = self.fc2(p).squeeze(2).sigmoid()
#         return p

#     def gat_layer(self, x, p, mask=None):
#         for m in self.layers:
#             x, p = m(x, p, mask)
#         return x, p

#     def entity_trans(self, x, pos):
#         e1 = x * pos.unsqueeze(2).expand(-1, -1, x.size(2))
#         # avg
#         if self.config.pool_type == 'avg':
#             divied = torch.sum(pos, 1)
#             e1 = torch.sum(e1, 1) / divied.unsqueeze(1)
#         elif self.config.pool_type == 'max':
#             # max
#             e1, _ = torch.max(e1, 1)
#         return e1


# class GATLayer(nn.Module):

#     def __init__(self, hidden_size):
#         super().__init__()
#         self.ra1 = RelationAttention(hidden_size)
#         self.ra2 = RelationAttention(hidden_size)

#     def forward(self, x, p, mask=None):
#         x = self.ra1(x, p) + x
#         p = self.ra2(p, x, mask) + p
#         return x, p

# class RelationAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(RelationAttention, self).__init__()
#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.score = nn.Linear(2*hidden_size, 1)
#         self.gate = nn.Linear(hidden_size * 2, 1)

#     def forward(self, p, x, mask=None):
#         q = self.query(p)#bcd
#         k = self.key(x)#bld
#         score = self.fuse(q, k)#bcl
#         if mask is not None:
#             mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
#             score = score.masked_fill(mask == 1, -1e9)
#         score = F.softmax(score, 2)
#         v = self.value(x)
#         out = torch.einsum('bcl,bld->bcd', score, v) + p
#         g = self.gate(torch.cat([out, p], 2)).sigmoid()
#         out = g * out + (1 - g) * p
#         return out

#     def fuse(self, x, y):
#         x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
#         y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
#         temp = torch.cat([x, y], 3)
#         return self.score(temp).squeeze(3)

    # def fuse(self, x, y):
    #     out = torch.einsum('bcl,bld->bcd', x, y.transpose(1,2))/math.sqrt(x.size(-1))
    #     return out


# class RelationFilm(nn.Module):
#     def __init__(self, hidden_size):
#         super(RelationFilm, self).__init__()
#         self.query = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.key = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.score = nn.Linear(2 * hidden_size, 1)
#         self.gate = nn.Linear(hidden_size * 2, 1)
#
#     def forward(self, p, x, mask=None):
#         out = self.fuse(p, x, mask)+x
#         g = self.gate(torch.cat([out, x], 2)).sigmoid()
#         out = g * out
#         return out
#     def fuse(self, x, y, mask=None):
#         a = self.query(x)#bcd
#         y = self.key(y)
#         a = a.unsqueeze(1).expand(-1, y.size(1), -1, -1)
#         y = y.unsqueeze(2).expand(-1, -1, x.size(1), -1)
#         y = a * y#blcd
#
#         if mask is not None:
#             mask = 1 - mask[:, None, :, None].expand(-1, y.size(1), -1, y.size(3))
#             y = y.masked_fill(mask == 1, 0)
#
#         out = torch.tanh(y).sum(2)
#         return out