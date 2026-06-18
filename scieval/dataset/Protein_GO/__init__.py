"""
Protein GO (Gene Ontology) dataset for SciEvalKit.

Supports go_bp, go_cc, go_mf - protein GO annotation prediction.
Data: LMUDataRoot()/go_bp.tsv, go_cc.tsv, go_mf.tsv
Mappings: Protein_GO/mappings/go_bp_mapping.txt, go_cc_mapping.txt, go_mf_mapping.txt
"""
from .protein_go import ProteinGODataset

__all__ = ["ProteinGODataset"]
