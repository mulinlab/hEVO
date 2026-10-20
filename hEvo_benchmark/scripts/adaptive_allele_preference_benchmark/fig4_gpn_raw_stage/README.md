# Fig. 4f GPN-Star raw-score preparation

The standard GPN-Star VEP path exports an alt/ref LLR. Fig. 4f's
Calinski–Harabasz panel uses two raw score vectors per model: the 30
selected-allele scores and the 30 other-allele scores.

This stage uses `gpn.star.logits.LogitsInference`, preserves the full
`- A C G T ?` vocabulary, maps reverse-complement logits to forward
orientation, averages forward/reverse logits, and applies full-vocabulary
log-softmax before extracting `selected_logprob` and `other_logprob`.

The resulting rows are merged into S39, where the Fig. 4f plotting script reads
the selected- and other-allele score vectors to compute CH values.

## GPN-MSA raw-score repair

`gpn_msa_raw_logits_adapter.py` uses the published GPN-MSA masked-LM model
with the multiz100way Zarr v2 alignment.  It extracts full-vocabulary
log-softmax values at the masked variant site in both forward and reverse
orientations, averages the two orientations for each allele, and writes the
30 selected/other raw-score pairs required by the Fig. 4f CH calculation.
`merge_fig4_gpn_msa_raw_scores.py` adds the GPN-MSA score vectors to S38/S39
while preserving the rows for other models.

[Back to parent guide](../../README.md)
