#!/usr/bin/env python3
"""
Memory Optimization Example

This example demonstrates various techniques for optimizing memory usage
when working with the Unified DTA System.
"""

import sys
import os
import time
import torch
import gc
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from unified_dta import UnifiedDTAModel
from unified_dta.utils import MemoryMonitor

def create_test_data(n_samples: int = 100) -> Tuple[List[str], List[str]]:
    """Create test data for memory optimization demonstrations."""
    import random
    
    # Sample molecules
    molecules = ["CCO", "CC(=O)O", "CC(C)O", "C1=CC=CC=C1", "CC(C)(C)O"]
    protein = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    smiles_list = [random.choice(molecules) for _ in range(n_samples)]
    protein_list = [protein] * n_samples
    
    return smiles_list, protein_list

def demonstrate_batch_size_optimization():
    """Show how batch size affects memory usage."""
    print("=== Batch Size Optimization ===\n")
    
    # Create model
    config = {
        'model': {'protein_encoder_type': 'cnn', 'drug_encoder_type': 'gin', 'use_fusion': False},
        'protein_config': {'output_dim': 64},
        'drug_config': {'output_dim': 64, 'num_layers': 3},
        'predictor_config': {'hidden_dims': [128]}
    }
    model = UnifiedDTAModel.from_config(config)
    
    # Test data
    smiles_list, protein_list = create_test_data(200)
    
    batch_sizes = [1, 4, 8, 16, 32]
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        with MemoryMonitor() as monitor:
            # Proces   main()n__":
 "__mai__ == _namef _)

id! ==="tecompleon example timizatiy opor Mem==t("\n= 
    prins.")
   iencng dependemissior memory fficient e to insumight be dus rint("Thi)
        p{e}"n: nstratiodemog r durinrint(f"Erro
        p e:on as Exceptiept   exc
        
 ing()or_monitrye_memo  demonstrat
      ()timizationize_oprate_model_sdemonst       ()
 sing_procesntmory_efficietrate_me  demons
      ion()izatoptimh_size__batcemonstrate      d   try:
  t()
    
  prin
      GB")
  emory:.1f} {system_mAvailable:Memory m Systent(f"       pri1e9
 / total al_memory().il.virtuutory = psystem_mem       sutil
    import pselse:
     B")
    :.1f} Gpu_memorylable: {gvaiory At(f"GPU Mem      prinry / 1e9
  l_memos(0).totaropertiece_pevi.cuda.get_dmory = torch    gpu_me  le():
  ilabs_avauda.itorch.c    if 
 memoryheck system   # C    
 ===\n")
ple Examimization Opty stem - MemorDTA Synified rint("=== Uin():
    pdef ma)

ts"arge datase very lsing foroces CPU prsidernt("6. Con)
    priryMonitor"with Memoory usage r memtont("5. Moni")
    pririodicallye pect garbag and collear cacheleint("4. C)
    prs"onuratinfigt model coe lightweighint("3. Us")
    prn chunks irge datasetsocess laPr. "2    print()
ry"ed memoor limitch sizes fsmaller bat("1. Use int)
    pr"tion Tips:mizapti Oory Mem\n💡 print(" tips
   y usageMemor  
    # nup")
  ter cleaasure("Afor.me)
    monitmpty_cache(ch.cuda.e       tor():
 _availableh.cuda.istorc   if )
 ollect( gc.cictions
   el pred d   variables
ar 
    # Clen")
    ioredict"After pasure(monitor.me    oteins)
 test_prmiles,_sbatch(testredict_odel.pctions = mredi pictions
   Make pred 
    # ding")
   ter data loasure("Af monitor.mea
    # Load data 
   tial")
   ("Inieasurer.m
    monitor()
    nitoMoiledMemorynitor = Deta  
    mo
  1f} MB"):.mbmemory_}: {abel {lnt(f" ri    p      ))
  _mbbel, memorynd((laments.appesuremeaf.        sel
               024
 1024 / 1s / o().rsemory_infProcess().m = psutil._mb   memory         
    sutilort p     imp          
      else:      1024
  1024 /) / ocated(mory_allda.meorch.cumory_mb = tme             
   lable():avai.is_h.cuda   if torc         r):
l: st, labere(selfeasu    def m     
    
   ents = []suremself.mea           ):
 _init__(self   def _   Monitor:
  tailedMemory  class De
  
    iction:")uring predge dusaemory  print("M
   
    t_data(100) create_tesins =est_protees, test_smil   t  
 nfig)
  g(coom_confiDTAModel.frUnifiedel =   }
    mod
  }[128]dims': ': {'hidden_r_configredicto 'p       yers': 3},
_lam': 64, 'num_di'output_config': {drug        'im': 64},
utput_dig': {'oonfotein_c  'pre},
      ': Falsionn', 'use_fusype': 'gider_tdrug_enco 'cnn', 'type':in_encoder_: {'proteodel''m       
  = {   config")
    
 ent ===\ngemManad itoring anon Memory M("\n==="
    print"" usage.oryge memor and manamonitw to ""Show ho
    "):ing(onitory_mmornstrate_meef demo
dd: {e}")
Faileme:<8} rint(f"{na         p    as e:
ionpt except Exce 
               }")
   e:<10.1f_tim1f} {predory_mb:<12..peak_memtor.1f} {moni<10} {size_mb:{params:<10,} name:<8t(f"{    prin       
     0
        * 100time) t_ime() - star.t = (timemeed_ti   pr        
     proteins)est__smiles, tch(testatedict_bel.prod   _ = m             ()
 = time.timetimert_   sta          
   itor: mon) asryMonitor(ith Memo w              
         1024
 4 / 1024 / = params *e_mb  siz
           ))parameters(n model.for p imel() sum(p.nu params =       ig)
     base_conf_config(.fromelnifiedDTAMod   model = U         :
 try           
  verride)
  config_odate(se_config.up
        ba }   lse}
    fusion': Fase_ 'gin', 'u_type':g_encoder', 'drunntype': 'coder_n_encotei {'prl':    'mode     g = {
   base_confi       ms():
 ite in configs.deig_overrionfme, c    for na
  " * 60)
  print("-")
    0}e (ms)':<1':<12} {'Timry (MB)} {'Memo10MB)':<'Size (s':<10} {amar:<8} {'P"{'Model'(fprint
     ata(50)
   _dcreate_testteins = st_pro, test_smileste  
       }
  }
 
        [256, 128]}dims': 'hidden_': {r_config  'predicto
          },yers': 48, 'num_ladim': 12output__config': {' 'drug     8},
      : 12ut_dim'g': {'outpein_confi  'prot
          {ge':   'Lar         },
8]}
     s': [12'hidden_dim': {ctor_config   'predi        s': 3},
 'num_layer64, _dim':  {'outputonfig': 'drug_c          },
 dim': 64ut_ig': {'outptein_conf'pro          {
   'Medium':     ,
      }4]}
     ': [6dden_dims{'hi_config': or'predict             2},
ayers':'num_l 32, ':{'output_dimfig':    'drug_con
         ': 32},'output_dimg': {tein_confiro   'p    l': {
      'Smal   ,
    }
        }dims': [32]den_'hidfig': {dictor_conpre           '': 1},
 rsm_laye': 16, 'nuput_dim: {'out_config'ugdr           '16},
 put_dim':  {'out':config 'protein_        iny': {
   'T      
  gs = {confi  
    
  n")zation ===\OptimiModel Size \n===  print("""
   ."emory usage mand theirizes  sent modelpare differ""Com"    zation():
ptimimodel_size_oe_f demonstrat
de.1f}%")
* 100:] - 1) memory_mb'0]['s[esult rmb /k_memory_or.pea(monituction: {emory rednt(f"   Mri")
    pBry_mb:.1f} Mk_memoor.pea: {monitMemoryf"   
    print(ect()
    gc.coll             
   ne() else No_availableuda.is) if torch.cempty_cache(da.orch.cu        t:
        ) == 0e * 5unk_sizf i % (ch  i     y
     riodicallche per caea      # Cl            
      preds)
nk_s.extend(chuionnked_predict       chu   teins)
  s, chunk_pronk_smiletch(chubal.predict_preds = mode     chunk_ze]
       hunk_siist[i:i+cprotein_lteins = chunk_pro            nk_size]
i:i+chues_list[es = smilmil   chunk_s:
         _size) chunk),iles_list, len(smn range(0 i i
        foronitor:tor() as moryMonith Mem  wi    
  ions = []
_predicthunked c  0
 k_size = 5  chuning:")
  esshunked proc"\n2. Cint(    
    pr} MB")
b:.1fk_memory_mea: {monitor.pMemorynt(f"     prilist)
  rotein_es_list, psmilch(edict_bat = model.prpredictionsr:
        onito) as moryMonitor(  with Memg:")
  rocessinard p"1. Standt(
    prin 
   0)a(100e_test_dat creatn_list =list, protei    smiles_aset
 datge # Lar
    
   (config)figrom_condDTAModel.fifieodel = Un   m
    }64]}
 ims': [ {'hidden_dnfig':tor_copredic
        'yers': 2},32, 'num_laim': 'output_d: {_config'drug     '
   32},utput_dim': fig': {'oin_conotepr
        '': False},se_fusionn', 'upe': 'gioder_tyencn', 'drug_: 'cnpe'_encoder_ty{'protein  'model':      onfig = {
 model
    cghtweight reate li  # C  n")
    
==\cessing =Proy-Efficient \n=== Memorrint(""""
    pechniques.essing tent procici-effmory""Show meg():
    "sin_procescientmory_effistrate_me
def demon:.2f}s)")
['time_s']ptimal {otime:']:.1f} MB, ['memory_mbimaly: {opt (memorch_size']}al['battim: {oph size batcOptimal💡  print(f"
   x['time_s'])_mb'] * mory x: x['me key=lambdaults,l = min(res
    optimasizeh mal batc# Find opti
    
    ed/s\n")f} prme:.1al_tis) / totn(predictionut: {leoughpint(f"  Thr      pr")
  s2f}_time:.Time: {total     print(f"
     } MB")y_mb:.1fmor.peak_memonitor {f"  Memory:     print(   
   
           })me
  / total_tions) ctipredien(hput': l'throug     
       time,al_s': tot 'time_          
 b,_memory_mor.peakmb': monitemory_          'm  tch_size,
 bae':ch_siz        'batnd({
    s.apperesult        
 
        start_time -ime.time()tal_time = t   to
                  _preds)
   extend(batchictions.  pred              ins)
batch_protetch_smiles, ct_batch(ba model.prediatch_preds =       b   
      ze]:i+batch_siin_list[i= protes tch_protein   ba        ize]
     i:i+batch_sst[li= smiles_tch_smiles      ba          size):
 atch_, bs_list)smilege(0, len(an  for i in r     
         ()
        .timee = time  start_tim
           = []ons    predicti       chunks
 s in 