import torch
from torchdrug.data import Molecule, PackedProtein
from torchdrug import core
from peer import util
from script.run_single import build_solver  # Adjust path if needed

# --- Dummy logger ---
class DummyLogger:
    def warning(self, msg): pass
    def info(self, msg): pass
    def debug(self, msg): pass

# --- Inference function ---
def run_inference(model, smiles_list, protein_list, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    
    from itertools import product
    all_pairs = list(product(smiles_list, protein_list))  # (ligand, protein) pairs
    
    predictions = []
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i+batch_size]
        
        ligands = [Molecule.from_smiles(pair[0]) for pair in batch_pairs]
        proteins = [PackedProtein.from_sequence(pair[1]) for pair in batch_pairs]
        
        batch_ligand = Molecule.pack(ligands).to(device)
        batch_protein = PackedProtein.pack(proteins).to(device)
        
        batch = {"graph1": batch_protein, "graph2": batch_ligand}
        
        with torch.no_grad():
            batch_pred = model.predict(batch)
        
        predictions.extend(batch_pred.cpu().tolist())  # convert to list if tensor
    
    # Reshape to (N, M)
    import numpy as np
    preds_array = np.array(predictions).reshape(len(smiles_list), len(protein_list))
    
    return preds_array
    
# --- Load config and checkpoint ---
cfg = util.load_config("/home/amberlab/scratch/torchprotein_output/InteractionPrediction/BindingDB/ProteinConvolutionalNetwork_2025-08-05-10-08-50/bindingdb_CNN.yaml")
cfg.checkpoint = "/home/amberlab/scratch/torchprotein_output/InteractionPrediction/BindingDB/ProteinConvolutionalNetwork_2025-08-05-10-08-50/model_epoch_90.pth"

# --- Build solver & extract model ---
solver = build_solver(cfg, DummyLogger())
model = solver.model

# --- Run inference ---
smiles = [
    "OC[C@@H](O1)[C@H](O)[C@H](O)[C@@H](O)[C@H]1-OCCCN-C(=O)CCCOC", 
    "OC[C@@H](O1)[C@H](O)[C@H](O)[C@@H](O)[C@H]1-OCCCN-C(=O)CCCOC"]
sequence = [
    "MMRMRHTAISLLALALFFLKVSAKLSLPFYLPANETLGLEVGNTSAQPYSEERCGIQAGGRRCPNGMCCSYTGWCGNTSKHCNPNNCQSQCSGPFPRDRCGWQADGRSCPTGVCCSECGWCGTTSAYCAPSNCQDQCEKLPSPSPPPPSPPPPPPSPSPPPPPPSPPPPPSYPQGRCGRQAGGRKCPTGVCCSLSGWCGTTSAYCNPDICQSQCSGPFPQGRCGWQADGGVCPTGVCCSLSGWCGTTSAYCAPGHCQSQCPTLIKNGIRGIESFLLNAI"
]

prediction = run_inference(model, smiles, sequence)
print(prediction)
