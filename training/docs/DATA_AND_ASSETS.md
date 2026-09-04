# Training Data and Assets

The six training, validation, and test workbooks are under
[`data_and_benchmarks/01_Datasets`](../../data_and_benchmarks/01_Datasets/README.md),
and the released SAAFEC-STAIR checkpoints are under
[`inference_checkpoints/`](../../inference_checkpoints/README.md).

The following files must be supplied separately:

- wild-type PDB or modeled structures;
- SaProt and ProteinMPNN weights;
- Foldseek; and
- the pinned ProteinMPNN utility from ThermoMPNN.

Set their paths with the environment variables or command-line arguments in
the [training README](../README.md).

## Structure Records

The exact wild-type ColabFold structures used for the reported SaProt and
ProteinMPNN features are not included. Reproducing those features requires the
same structures and a manifest containing:

- protein identifier and dataset split;
- structure filename and chain;
- amino-acid sequence or sequence checksum;
- SHA-256 checksum of each PDB file; and
- ColabFold version, model settings, database snapshot, and rank-selection rule.

If the structures are regenerated, record the sequences, commands,
configuration, software version, database snapshot, selection rule, and
checksums. Different ColabFold versions or database snapshots may produce
different structures.

See the repository guides for [external assets](../../docs/EXTERNAL_ASSETS.md),
[checkpoints](../../docs/CHECKPOINTS.md), and
[third-party licenses](../../docs/THIRD_PARTY_LICENSES_AND_NOTICES.md).

Use a tagged release or commit when citing the workbooks. Exact reproduction
also requires the structure files and their checksum manifest.

## Double-Mutation Dependency

The double-mutation workflow loads the three Tier 2 single-mutation checkpoints
from each seed's `phase1/checkpoints/best_head.pt`, keeps them frozen, and
trains the pair-correction head. The released single-mutation checkpoints can
be used for this stage. The released double-mutation checkpoint is for
inference and is not a training input.
