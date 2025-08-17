#!/usr/bin/env python3
"""
Baseline Comparison Example

This example demonstrates how to compare the Unified DTA System
with baseline models and evaluate performance.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unified_dta import UnifiedDTAModel

def create_synthetic_dataset(n_samples: int = 200) -> Tuple[List[str], List[str], List[float]]:
    """Create a synthetic dataset for comparison."""
    import random
    
    # Sample molecules with different properties
    molecules = [
        "CCO",                    # Small, polar
        "CC(=O)O",               # Small, acidic
        "CC(C)O",                # Small, branched
        "C1=CC=CC=C1",           # Aromatic
        "CC(C)(C)O",             # Bulky
        "CCCCCCCCCCCCCCCCCC(=O)O", # Long chain
        "C1=CC=C(C=C1)O",        # Phenolic
        "CC(=O)N",               # Amide
        "CCCCO",                 # Linear alcohol
        "C1CCCCC1"               # Cyclic
    ]
    
    # Sample protein (kinase domain)
    protein_sequence = (
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        "RHPKMWVGVLLFRIGGGSSVGAGTTMGKSTTSAAITAACWSRDVLKKNKHVDGCMYEEQNLSV"
        "IRGSIAHAIYLNTLTNMDGTERELLESYIDGRRLVRGDGSFAKLVRPCSLDYVAIHGFLTNYH"
    )
    
    # Generate random combinations
    np.random.seed(42)
    smiles_list = np.random.choice(molecules, n_samples).tolist()
    protein_list = [protein_sequence] * n_samples
    
    # Generate syn"===     print(""
set."datae s on the samerent modeliffe d""Compar
    "s():model compare_

def }  arman_p
 rman_P': speSpea   ',
      spearman_rarman_R':       'Spen_p,
 earso prson_P':   'Pea_r,
     arson: pe'Pearson_R'
        ': mae,  'MAEe,
      ': rmsMSE
        'RMSE': mse,        '  return {

  )
    true_valuesictions, r(predspearmanrman_p = n_r, speaearmasp    values)
e_s, truictionsonr(predson_p = pearson_r, pearear   ps
 ion metriclat  # Corre
      tions)
ic predalues,ror(true_vbsolute_er = mean_amae
    rt(mse).sq= npe msions)
    redict prvalues,error(true__squared_  mse = meanics
  trgression me Re    #   
es)
 true_valunp.array(s = value    true_ns)
iodict(preraynp.arctions = 
    predi""ics."metrltiple e with muperformancluate model "Eva
    "" float]:[str,t]) -> Dictfloas: List[, true_valuefloat]ns: List[l(predictio_modeateef evalu()

dist)).tolistn(smiles_ltd, lean, self.sal(self.meom.normrn np.rand        retu"""
ons.redictirandom p""Make "    
    :oat]) -> List[flt: List[str]rotein_lisst[str], plist: Lies_millf, sct(se   def predi
    
 es)(affiniti np.stdlf.std =
        senities)fip.mean(af.mean = n self""
       aseline." random b"Fit the       ""[float]):
 ties: Listtr], affinit: List[s, smiles_lisfit(self    def    
 = 1.5
 elf.std
        s 5.0elf.mean = s  
     :elf)__(s_initdef _  
    """
  comparison. for baselineandom   """R  mBaseline:
s Rando
clasdictions
  return pre   
      ed)
     prons.append(   predicti      
   0.2).normal(0, andomnp.r=       pred +      se
 some noi Add       #  
           .1
       pred -= 0           
  dsuble bons:  # Do smile=" in "        if  15
   pred += 0.       
        miles:" in s   if "N1
          pred += 0.         :
      in smiles" "Of         i  
  oupsnal grtiod on funcdjust base         # A
               -= 0.3
pred             5:
    iles) > 1 elif len(sm           red += 0.2
     p           
< 5:(smiles)   if len    
      cular size on moleed# Adjust bas                   

     finityf.mean_afd = sel        pre   uristics
  # Simple he     t:
      smiles_lisles in mi      for s   
  = []
     ictions       pred  ""
ies."opertlar prolecun simple mons based opredicti""Make    "     
loat]:t[f-> Lis) str]st[Li: rotein_liststr], plist: List[s_lef, smiredict(sel
    def pes)
    nitip.mean(affi= nn_affinity elf.mea  s    """
  ine model.it the basel"""F        float]):
ies: List[ affinit],str_list: List[lessmiit(self, def f   
     5.0
     finity =lf.mean_af   se:
     t__(self)def __ini   
 ""
    roperties."r p on moleculaasedpredicts bel that e modelinple basim""A s:
    "lineimpleBaselass Sities

cffin, a_listeinprotist, s_lturn smile re    
   nity)
_affiappend(baseaffinities.  
        .0)
      al(-0.2, 1om.normand= np.rinity +se_aff    ba    
    r moleculesthese:  # O    el
    l(0.5, 0.8)random.normanp.+= affinity    base_s
         molecule # Polar s: in smileO" "if  el.5)
       normal(0, 0= np.random.ffinity +  base_a        s
  eculell mol5:  # Smamiles) <   if len(s  rties
    opeolecule prased on m structure bdd some       # A
 inity = 5.0ff     base_a  ist:
 n smiles_lfor smiles i= []
    affinities    ity
 inr aff have loweend toolecules tler m Smalre
    # structues with somenitifitic afthe