import torch
import torch.nn as nn
from supar.modules.lstm import LSTM
from supar.modules.affine import Biaffine
from supar.modules.bert import BertEmbedding
from supar.modules.mlp import MLP
from supar.modules.scalar_mix import ScalarMix
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.utils import Config
from supar.modules.char_lstm import CharLSTM
from supar.utils.alg import eisner, eisner2o, mst
from supar.utils.transform import CoNLL
from supar.models.dependency import BiaffineDependencyModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TwoThread(nn.Module):
    def __init__(self,
                 n_words,
                 n_feats,
                 n_rels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 lexemetrain=False,
                 postrain=False,
                 **kwargs):
        super().__init__()
        self.args = Config().update(locals())

        self.lexemeThread = BiaffineDependencyModel(
                n_words,
                n_feats,
                n_rels,
                feat,
                n_embed,
                n_feat_embed,
                n_char_embed,
                bert,
                n_bert_layers,
                mix_dropout,
                embed_dropout,
                n_lstm_hidden,
                n_lstm_layers,
                lstm_dropout,
                n_mlp_arc,
                n_mlp_rel,
                mlp_dropout,
                feat_pad_index,
                pad_index,
                unk_index,
                **kwargs
        )
        
        n_words2 = kwargs.get("n_words2", n_words)
        n_feats2 = kwargs.get("n_feats2", n_feats)
        n_rel2 = kwargs.get("n_rel2", n_words)
        feat2 = kwargs.get("feat2", feat)
        n_embed2 = kwargs.get("n_embed2", n_embed)
        n_feat_embed2 = kwargs.get("n_feat_embed2", n_feat_embed)
        n_char_embed2 = kwargs.get("n_char_embed2", n_char_embed)
        bert2 = kwargs.get("bert2", bert)
        n_bert_layers2 = kwargs.get("n_bert_layer2", n_bert_layers)
        mix_dropout2 = kwargs.get("mix_dropout2", mix_dropout)
        embed_dropout2 = kwargs.get("embed_dropout2", embed_dropout)
        n_lstm_hidden2 = kwargs.get("n_lstm_hidden2", n_lstm_hidden)
        n_lstm_layers2 = kwargs.get("n_lstm_layers2", n_lstm_layers)
        lstm_dropout2 = kwargs.get("lstm_dropout2", lstm_dropout)
        n_mlp_arc2 = kwargs.get("n_mlp_arc2", n_mlp_arc)
        n_mlp_rel2 = kwargs.get("n_mlp_rel2", n_mlp_rel)
        mlp_dropout2 = kwargs.get("mlp_dropout2", mlp_dropout)
        feat_pad_index2 = kwargs.get("feat_pad_index2", feat_pad_index)
        pad_index2 = kwargs.get("pad_index2", pad_index)
        unk_index2 = kwargs.get("unk_index2", unk_index)

        self.posThread = BiaffineDependencyModel(
                n_words2,
                n_feats2,
                n_rels2,
                feat2,
                n_embed2,
                n_feat_embed2,
                n_char_embed2,
                bert2,
                n_bert_layers2,
                mix_dropout2,
                embed_dropout2,
                n_lstm_hidden2,
                n_lstm_layers2,
                lstm_dropout2,
                n_mlp_arc2,
                n_mlp_rel2,
                mlp_dropout2,
                feat_pad_index2,
                pad_index2,
                unk_index2,
                **kwargs
        )

        self.lexemetrain = lexemetrain
        self.postrain = postrain
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def freeze_unfreeze(self):
        for p in self.lexemeThread.parameters():
            p.requires_grad = lexemetrain
        
        for p in self.posThread.parameters():
            p.requires_grad = postrain
    
    def load_pretrained(self, embed=None, embed2=None):
        if embed is not None:
            #self.pretrained = nn.Embedding.from_pretrained(embed)
            self.lexemeThread.load_pretrained(embed)
            #self.posThread.load_pretrained(embed)
        if embed2 is not None:
            self.posThread.load_pretrained(embed2)
        return self

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
            for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
        
    def loss(self, s_arc, s_rel, arcs, rels, mask, partial=False):

        if partial:
            mask = mask & arcs.ge(0)
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss
                
        
    def forward(self, words, feats, tags):
        if self.lexemetrain:
            a_arc, a_rel = self.lexemeThread(words, feats)
        if self.postrain:
            b_arc, b_rel = self.posThread(tags, feats)
        if self.postrain and self.lexemetrain:
            x_arc = a_arc * b_arc
            x_rel = a_rel * b_rel
        elif self.postrain:
            x_arc = b_arc
            x_rel = b_rel
        elif self.lexemetrain:
            x_arc = a_arc
            x_rel = a_rel
        return x_arc, x_rel
