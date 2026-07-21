# External Assets

Third-party code, executables, and weights are not distributed in this
repository. Obtain each item from its official source, review its license, and
place it at the expected ignored path.

| Component | Expected path | Upstream license |
| --- | --- | --- |
| SaProt 650M PDB | `assets/external/models/SaProt_650M_PDB/` | MIT |
| ProteinMPNN `v_48_020.pt` | `assets/external/models/proteinmpnn/v_48_020.pt` | MIT |
| ThermoMPNN ProteinMPNN utility | `assets/external/source/ThermoMPNN/protein_mpnn_utils.py` | MIT |
| Foldseek executable | `assets/external/bin/foldseek` | GPL-3.0 |

License labels summarize the upstream repositories as inspected for this
release. The upstream license and terms remain authoritative.

See [Third-party licenses and notices](THIRD_PARTY_LICENSES_AND_NOTICES.md)
for the documentation credit summary.

## SaProt

Official sources:

- Model: <https://huggingface.co/westlake-repl/SaProt_650M_PDB>
- Code: <https://github.com/westlake-repl/SaProt>

The expected model directory contains at least:

```text
assets/external/models/SaProt_650M_PDB/
  config.json
  pytorch_model.bin
  special_tokens_map.json
  tokenizer_config.json
  vocab.txt
```

## ProteinMPNN

Official source:

- <https://github.com/dauparas/ProteinMPNN>

Restore the official vanilla model weight as:

```text
assets/external/models/proteinmpnn/v_48_020.pt
```

## ProteinMPNN Utility Attribution

SAAFEC-STAIR loads the `ProteinMPNN` class at runtime from ThermoMPNN's
`protein_mpnn_utils.py`. Use the file at pinned ThermoMPNN commit:

```text
13569795daa7689b6a6df0279b383e08c6212e79
```

Upstream location:

<https://github.com/Kuhlman-Lab/ThermoMPNN/blob/13569795daa7689b6a6df0279b383e08c6212e79/protein_mpnn_utils.py>

Save it as:

```text
assets/external/source/ThermoMPNN/protein_mpnn_utils.py
```

Expected SHA-256:

```text
3bbcb4342482438bb5d4ebe6509d514490dfce804617865fa55ffdcbda2fea12
```

This module supplies the ProteinMPNN architecture implementation used to load
the official `v_48_020.pt` weight. ThermoMPNN itself is not run as a predictor
in this pipeline.

## Foldseek

Official sources:

- Releases: <https://github.com/steineggerlab/foldseek/releases>
- Code: <https://github.com/steineggerlab/foldseek>

For Linux, use the official SSE4.1 build as the compatibility-first default:

```bash
curl -L https://mmseqs.com/foldseek/foldseek-linux-sse41.tar.gz \
  -o foldseek-linux-sse41.tar.gz
tar -xzf foldseek-linux-sse41.tar.gz
```

The official AVX2 build may be used on CPUs that support AVX2. Running an AVX2
binary on an incompatible CPU can terminate with an `Illegal instruction`
error.

Install a compatible official executable at:

```text
assets/external/bin/foldseek
```

Make it executable and verify it:

```bash
chmod +x assets/external/bin/foldseek
assets/external/bin/foldseek version
```

Foldseek is used to create SaProt's structure-aware sequence representation.

## Alternate Locations

External assets may remain in centrally managed installations:

```bash
SAPROT_MODEL_DIR=/shared/models/SaProt_650M_PDB \
FOLDSEEK_BIN=/shared/bin/foldseek \
PROTEINMPNN_CHECKPOINT=/shared/models/proteinmpnn/v_48_020.pt \
PROTEINMPNN_SOURCE=/shared/src/ThermoMPNN/protein_mpnn_utils.py \
MODE=single \
bash run_saafec_stair_inference.sh
```
