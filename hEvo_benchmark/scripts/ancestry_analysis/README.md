# Ancestry Analysis

These scripts describe the optional PLINK/ADMIXTURE analysis used to compare learned sequence embeddings with conventional ancestry estimates.

Set the ancestry input and output paths when running the workflow:

```bash
bash plink_admixture_workflow.sh \
  /path/to/phased_or_hgdp_vcf_dir \
  hEvo_benchmark/results/ancestry_analysis \
  5
```

The workflow requires `plink`, GNU `parallel`, and `admixture` in the active environment.

[Back to parent guide](../README.md)
