import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

__all__ = ['MULT']

class MULT(nn.Module):
    
    def __init__(self, args):

        super(MULT, self).__init__()
        
        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir = args.cache_path)


        video_feat_dim = 768
        text_feat_dim = 768
        audio_feat_dim = 768 * 4

        dst_feature_dims = args.dst_feature_dims

        rnn = nn.LSTM

        self.arnn1 = rnn(768, 768, bidirectional=True)
        self.arnn2 = rnn(2 * 768, 768, bidirectional=True)

        self.alayer_norm = nn.LayerNorm((768 * 2,))

        self.orig_d_l, self.orig_d_a, self.orig_d_v = text_feat_dim, audio_feat_dim, video_feat_dim
        self.d_l = self.d_a = self.d_v = dst_feature_dims

        self.num_heads = args.nheads
        self.layers = args.n_levels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        
        self.combined_dim = combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = args.num_labels   

        args.conv1d_kernel_size_l = 1
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        self.trans_l_with_a = self._get_network(self_type='la')
        self.trans_l_with_v = self._get_network(self_type='lv')

        self.trans_a_with_l = self._get_network(self_type='al')
        self.trans_a_with_v = self._get_network(self_type='av')

        self.trans_v_with_l = self._get_network(self_type='vl')
        self.trans_v_with_a = self._get_network(self_type='va')

        self.trans_l_mem = self._get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self._get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self._get_network(self_type='v_mem', layers=3)

        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
    def _get_network(self, self_type='l', layers=-1):

        if self_type in ['l', 'vl', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def _extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):

        packed_sequence = pack_padded_sequence(
            sequence, lengths, batch_first=True, enforce_sorted=False)

        packed_h1, (final_h1, _) = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(
            normed_h1, lengths, batch_first=True, enforce_sorted=False)

        _, (final_h2, _) = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def forward(self, text_feats, video_feats, audio_feats):
        batch_size = text_feats.size(0)
        lengths = (torch.ones(batch_size)*20).int()

        final_h1a, final_h2a = self._extract_features(
            audio_feats, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        audio_feats = torch.cat((final_h1a, final_h2a), dim=2).permute(
            1, 0, 2).contiguous().view(batch_size, -1)

        x_l = text_feats.unsqueeze(dim=2)
        x_a = audio_feats.unsqueeze(dim=2)
        x_v = video_feats.unsqueeze(dim=2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # (V,A) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)

        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim = 2)
        h_ls = self.trans_l_mem(h_ls)

        if type(h_ls) == tuple:
            h_ls = h_ls[0]

        last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        # (L,V) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]    

        # (L,A) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1) 
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs
        
        logits = self.out_layer(last_hs_proj)

        return logits, last_hs
     