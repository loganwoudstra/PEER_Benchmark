import torch
from torchdrug.data import Molecule, PackedProtein
from torchdrug import core
from peer import util
from itertools import product
import pandas as pd
from script.run_single import build_solver  # Adjust path if needed

# --- Dummy logger ---
class DummyLogger:
    def warning(self, msg): pass
    def info(self, msg): pass
    def debug(self, msg): pass

# --- Inference function ---
def run_inference(model, smiles_df, protein_df, batch_size=64):
    model.eval()
    device = next(model.parameters()).device

    # Create all pair combinations of ligands and proteins
    all_pairs = list(product(smiles_df.itertuples(index=False), protein_df.itertuples(index=False)))
    results = []

    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i + batch_size]

        ligands = [Molecule.from_smiles(ligand.SMILES) for ligand, _ in batch_pairs]
        proteins = [PackedProtein.from_sequence(protein.Sequence) for _, protein in batch_pairs]

        batch_ligand = Molecule.pack(ligands).to(device)
        batch_protein = PackedProtein.pack(proteins).to(device)
        batch = {"graph1": batch_protein, "graph2": batch_ligand}

        with torch.no_grad():
            batch_pred = model.predict(batch)

        batch_pred = [p.item() if torch.is_tensor(p) else p for p in batch_pred]
        results.extend([
            {
                "ligand_name": ligand.Name,
                "protein_name": protein.Name,
                "prediction": pred
            }
            for (ligand, protein), pred in zip(batch_pairs, batch_pred)
        ])

    return pd.DataFrame(results)
    
# --- Load config and checkpoint ---
cfg = util.load_config("/home/amberlab/scratch/torchprotein_output/InteractionPrediction/BindingDB/ProteinConvolutionalNetwork_2025-08-05-10-08-50/bindingdb_CNN.yaml")
cfg.checkpoint = "/home/amberlab/scratch/torchprotein_output/InteractionPrediction/BindingDB/ProteinConvolutionalNetwork_2025-08-05-10-08-50/model_epoch_100.pth"

# --- Build solver & extract model ---
solver = build_solver(cfg, DummyLogger())
model = solver.model

# --- Run inference ---
glycan_df = pd.read_csv('inference_data/select_glycans.txt', sep="\t")
protein_df = pd.read_csv('inference_data/select_proteins.txt', sep="\t")

prediction = run_inference(model, glycan_df, protein_df)
prediction.to_csv("inference_data/predictions.csv", index=False)
