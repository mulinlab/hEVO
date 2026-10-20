# Shapeit5 UK Biobank WGS phasing workflow

`shapeit5_phasing_pipeline.sh` is the early hEvo UK Biobank WGS phasing pipeline. `UKBB` remains the technical abbreviation used in script variables and input filenames. The workflow keeps the original nine-step order and checkpoint behavior while making all project-specific paths and tools configurable.

## Requirements

Run on HPC. The pipeline requires:

- PLINK2 with PGEN input support;
- bcftools and tabix;
- GNU parallel;
- GLIMPSE2 `chunk_static`;
- Shapeit5 `phase_common_static`, `phase_rare_static`, `ligate_static`, and `switch_static`;
- pandas in the HPC `llmbm` environment.

Every Python invocation in the pipeline is executed as:

```bash
micromamba \
  -r <micromamba-root> \
  -n llmbm run python ...
```

## Configuration

Required variables:

- `WORK_DIR`: pipeline work directory;
- `SHAPEIT5_DIR`: directory containing the four Shapeit5 binaries;
- `GLIMPSE2_CHUNK`: path to `GLIMPSE2_chunk_static`.

By default, the following are resolved relative to `WORK_DIR`:

- `UKBB_PGEN`: `${WORK_DIR}/input/chr${CHR}/chr${CHR}`;
- `GMAP_FILE`: `${WORK_DIR}/gmap_file/chr${CHR}.b38.gmap.gz`;
- `PED_FILE`: `${WORK_DIR}/ukb.related_column3.fam`;
- `BLOCK_DIR`: `${WORK_DIR}/chr_chunk_file_self`;
- `RESULT_DIR`: `${WORK_DIR}/plink_version_chr${CHR}_result`.

Override any of these with environment variables when the data layout differs. Tool variables (`PLINK2`, `BCFTOOLS`, `TABIX`, `PARALLEL`) accept either absolute executable paths or commands on `PATH`. `MICROMAMBA`, `MICROMAMBA_ROOT`, and `PYTHON_ENV` can be overridden, but the default Python environment is `llmbm`.

## Usage

```bash
cd <benchmark-root>

export WORK_DIR=/path/to/UKBB_WGS_phasing
export SHAPEIT5_DIR=/path/to/shapeit5-5.1.1
export GLIMPSE2_CHUNK=/path/to/GLIMPSE2_chunk_static

bash scripts/phasing/shapeit5_phasing_pipeline.sh 22
# Resume from a checkpoint:
bash scripts/phasing/shapeit5_phasing_pipeline.sh 22 step05
```

Use `bash scripts/phasing/shapeit5_phasing_pipeline.sh --help` to print the required configuration and step names. Before a run, the script checks the PGEN trio, genetic map, pedigree, all executable tools, Shapeit5 binaries, and the `llmbm` pandas import.

## Steps

1. Create 1 Mb chromosome blocks.
2. Convert and filter PGEN chunks to normalized VCF, then concatenate.
3. Index the merged VCF and create GLIMPSE2 chunks.
4. Convert GLIMPSE2 chunk records to rare/common chunk formats.
5. Phase common variants with Shapeit5.
6. Ligate common-variant phasing results.
7. Phase rare variants against the ligated scaffold.
8. Filter rare results by `AC >= MIN_AC_FILTER` and concatenate them.
9. Optional switch-error-rate preparation/measurement hook.

Default thresholds are: HWE `1e-30`, genotype missingness `0.1`, common MAF `0.01`, and final rare-variant AC `10`. Parallelism and thresholds can be overridden through the corresponding environment variables documented in the script.

Run the complete pipeline on HPC with the UKBB WGS inputs and phasing tools
listed above.

[Back to parent guide](../README.md)
