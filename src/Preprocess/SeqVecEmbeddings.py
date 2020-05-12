from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path
import torch
from Bio import SeqIO

model_dir = Path('../../Models/SeqVec/')
weights = model_dir / 'weights.hdf5'
options = model_dir / 'options.json'
embedder = ElmoEmbedder(options,weights, cuda_device=0)

output_dict = {}

for record in SeqIO.parse("../../Data/Fasta/huri_apid_proteins.fasta", "fasta"):
    id = record.id
    description = record.description
    seq = record.seq # your amino acid sequence
    embedding = embedder.embed_sentence(list(seq)) # List-of-Lists with shape [3,L,1024]
    protein_embd = torch.tensor(embedding).sum(dim=0).mean(dim=0) # Vector with shape [1024]
    print(protein_embd)
