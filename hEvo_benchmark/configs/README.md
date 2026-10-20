# Benchmark configuration

| Configuration | Purpose |
|---|---|
| [project.yaml](project.yaml) | Runtime input, output, cache and scheduler locations |
| [models.yaml](models.yaml) | Model identifiers, checkpoints and inference defaults |
| [tasks.yaml](tasks.yaml) | Dataset groups, input fields, labels and evaluation definitions |
| [gpn_star.yaml](gpn_star.yaml) | GPN-Star checkpoint and alignment configuration |

Set paths for the local environment before submitting jobs. Match model and
task identifiers to the relevant evaluator; display names and data definitions
are described in the [reader table guide](../tables/repository_tables/README.md).
The [evaluation guide](../scripts/evaluate/README.md) documents the shared
classification and paired-variant command-line interfaces.

[Back to parent guide](../README.md)
