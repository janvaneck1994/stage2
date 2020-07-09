import pandas as pd
from UniProtKBMapper import UniProtKBMapper
from pathlib import Path
from typing import List, Dict

def convert_interactome_to_uniprotKB(df: pd.DataFrame, mapping: List[str], mapping_output=None) -> pd.DataFrame:
    Mapper = UniProtKBMapper(df, mapping)
    if mapping_output:
        Mapper.save_mapping(mapping_output)
    return Mapper.get_converted()

# create paths
interactomes_dir = Path('../../Data/interactomes')
human_intact_dir = interactomes_dir / 'human'

### Interactomes
# Human
df_apid = pd.read_csv(human_intact_dir / 'external' / 'apid_human_2_no_int_species_04_05_2020.txt', sep='\t')
df_apid = df_apid[['UniprotID_A', 'UniprotID_B']]

df_huri = pd.read_csv(human_intact_dir / 'external' / 'HuRI_04_05_2020.psi', sep='\t', header=None)
df_huri = df_huri[[0,1]].applymap(lambda x: x.split(':')[-1].split('.')[0])

# convert ids
conv_huri = convert_interactome_to_uniprotKB(df_huri, ['ACC+ID','ENSEMBL_PRO_ID'], human_intact_dir / 'huri_mapping.csv')
conv_apid = convert_interactome_to_uniprotKB(df_apid, ['ACC+ID'], human_intact_dir / 'apid_mapping.csv')

# merge converted ids
df_human = pd.concat([conv_huri, conv_apid])
df_human = df_human.drop_duplicates()

# save interactome
df_human.to_csv(human_intact_dir / 'raw' / 'human_interactome.edgelist', index=False, header=False)
