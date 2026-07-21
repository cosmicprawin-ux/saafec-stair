# Input Templates

These CSV files are starting points for table-based inference. Copy a template
outside the repository, replace the example row, and provide the matching
wild-type PDB files with `INPUT_TABLE` and `PDB_DIR`.

## Single Mutations

`single_mutations_template.csv` contains one row per requested substitution:

- `pdb`: matching PDB filename, with or without `.pdb`.
- `chain`: protein chain used for prediction.
- `mutation`: compact substitution such as `A10V`.
- `position`: one-based position on the model sequence axis.
- `wt_aa` and `mut_aa`: one-letter wild-type and mutant amino-acid codes.

`INPUT_TABLE` selects proteins for full `L x 20` prediction matrices. Set
`SPECIFIED_MUTATION_TABLE` to the same file when an additional table containing
only the requested substitutions is wanted:

```bash
MODE=single \
INPUT_TABLE=/path/to/single_mutations.csv \
SPECIFIED_MUTATION_TABLE=/path/to/single_mutations.csv \
PDB_DIR=/path/to/pdbs \
bash run_saafec_stair_inference.sh
```

## Double Mutations

`double_mutations_template.csv` contains one requested mutation pair per row:

- `pdb`: matching PDB filename, with or without `.pdb`.
- `chain`: protein chain used for prediction.
- `mutation_1` and `mutation_2`: compact substitutions such as `A10V` and
  `G25D`.

The PDB filename may already include the chain suffix (for example, `1ABC_A`),
but it does not have to. The required `chain` column determines which chain is
used in either case.
