#!/usr/bin/env python3
"""
Custom Encoder Example

This example demonstrates how to create custom protein and drug encoders
and integrate them with the Unified DTA System.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unified_dta.core.base_components import BaseEncoder
from unified_dta import UnifiedDTAModel

class SimpleProteinEncoder(BaseEncoder):
    """A simple custom protein encoder using basic embeddings."""
    
    def __init__(self, vocab_size: int = 25, embed_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self._output_dim = output_dim
        
        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Forward pass for protein sequences."""
        # Simple tokenization (in practice, use proper tokenizer)
        tokenized = self._tokenize_sequences(sequences)
        
        # Embedding and pooling
        embedded = self.embedding(tokenized)  # [batch, seq_len, embed_dim]
        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)  # [batch, embed_dim]
        
        # Project to output dimension
        output = self.activation(self.projection(pooled))
        return output
    
    def _tokenize_sequences(self, sequences: List[str]) -> torch.Tensor:
        """Simple tokenization of protein sequences."""
        # Amino acid to index mapping
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        aa_to_idx['X'] = 20  # Unknown
        
        max_len = max(len(seq) for seq in sequences)
        batch_size = len(sequences)
        
        tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_len]):
                tokens[i, j] = aa_to_idx.get(aa, 20)
        
        return tokens
    
    @property
    def output_dim(self) -> int:
        return self._output_dim

class  main()   :
_main__"== "__name__ ")

if _ationgrtebefore inidually ers indivest encod T   print("-ry")
 l factoode mers with thecodr custom entegist("- Re")
    prinur encodersing in yo preprocesutdle inp("- Han)
    printroperty"_dim pd outputard() anorwement f- Impl  print("s")
  om encoderfor custEncoder m Baserit fro Inhet("-
    prinaways:")n💡 Key Take  print("\  

    )coders(tom_en_cusateonstr
    dem  )
  n"ple ===\ncoder Examm - Custom ETA Systenified D("=== U    print:

def main()")
ted! ===ple complexamom encoder eCust\n===  print("    
    data)")
phequires grady (rencoder rea("✓ Drug    print data)
 per graphrod p nee (wouldncoder erug  # Test d  
    
iled: {e}")er test faodncn e Proteint(f"✗
        pription as e:cept Exce
    ex.shape}")esatur {protein_feshape:der output rotein enco"✓ P  print(f
      roteins)est_per(totein_encod= prs rein_featuprote     
    try:
   DEFGHIKL"]", "ACKTVRQERLKs = ["Mrotein
    test_pcoder protein enTest
    # ")
    oders...encstom  custingTe"\n3. 
    print(iduallyders indiv. Test enco 
    # 3d")
   ateguration creder confiCustom encoint("✓   prion)
  ntegratctory istom fald need cuis woul (theate mode 
    # Cr
   
    }      }': 0.2
   'dropout         28],
  en_dims': [1       'hidd  
   onfig': {edictor_c   'pr        },
    
  Falseuse_fusion':      ',
      stom'r_type': 'cudrug_encode       '',
     : 'custom_type'in_encoder    'prote     ': {
   'model{
        ig =     confon
figuratie conimplcreate a swe'll , demo# For this 
    he system ters withcodese enregister thice, you'd : In pract# Note   
 )
    ."s..tom encoderh cusdel witing mo Creat\n2.rint("
    pders encoh customitn wnfiguratiol coCreate mode 2.  #    
   )
_dim}"outputder.rug_encoon: {dimensi dutOutpnt(f"   ri")
    p__}.__namelass__er.__ccodg_enoder: {drutom drug enc"✓ Cusprint(f}")
    _dimut.outpncodern_erotei: {pimensionutput drint(f"   O
    p")me__}_na_._er.__class_n_encodteirocoder: {protein en"✓ Custom p    print(f
    
m=64)tput_dir(ouEncodettentionDrugencoder = A   drug_
 =64)ut_dimr(outpinEncodeeProte Simplr =tein_encode)
    procoders..." custom enreatingrint("1. C
    psderom encostte cu Crea    # 1.)
    
===\n"r Example m Encodeustot("=== C   prin
 sage."""der uom encorate cust"Demonst""   ers():
 custom_encodate_demonstrim

def output_durn self._et        r) -> int:
put_dim(self    def outroperty
@p
    
    rn output     retued)
   f.ffn(poolel  output = s
      ward networkforFeed- #  
       
       dim]ut_inptch_size, m=1)  # [bamean(dittended.led = a        poos)
over noden oling (mea# Global po         
 zed)
      zed, normaliormaliormalized, nttention(n, _ = self.aded      attention
  # Self-atten               
)
 ded_features(padr_norm = self.layelized    norma   ization
 normalpply layer 
        # Ah
        n_batcodes_i)] = ne(0iz.schbatnodes_in_[i, :tures_fea    padded       sk]
 s[made_featuretch = nos_in_ba    node         == i
_indicesch batk =    mas
        h_size):n range(batc    for i i     
       .size(1))
featuress, node_max_nodech_size, ros(batorch.zefeatures = t  padded_
      ntion attee forshap re# Pad and 
        m()
       ).iteces).max(indi(batch_countbin= torch.  max_nodes + 1
      () .itemmax()indices.ch_ze = battch_si     ba
   by batchGroup nodes         #       

   [num_nodes]ch  #atch.bat graph_bs =ch_indice  bat
      put_dim]m_nodes, in# [nuch.x  graph_bateatures =  node_f       aph batch
rom grres fde featu no# Extract      
  ""r graphs."ulaecss for molard pa """Forw:
       h.Tensorrc> to -aph_batch)(self, grdef forward   
        
 nput_dim)(in.LayerNorm_norm = nf.layer       seltion
 r normalizaaye      # L      
      )
  )
    imm, output_dar(hidden_din.Line n           t(0.1),
   nn.Dropou   
        nn.ReLU(),        en_dim),
  t_dim, hiddear(inpuLin    nn.
        ential(nn.Sequf.ffn =    sel  work
    netFeed-forward     #        
      )
 
     uefirst=Tr   batch_        ,
 s=num_heads_head       num
     t_dim,ed_dim=inpu      emb
      ion(dAttenteaultihnn.Mtion = elf.atten        s attention
-headlti# Mu   
      im
       put_dutdim = o._output_      selfit__()
  __in   super().8):
      = 12im: intutput_dnt = 4, o iheads:       num_          
 : int = 128,, hidden_dimt = 78dim: int_puelf, in__init__(s def   
   ""
  anisms."n mechtioattener using ncodg edrutom ""A cus "oder):
   BaseEncer(nDrugEncodAttentio