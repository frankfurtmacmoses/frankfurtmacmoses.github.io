# Salmon RNA-seq Quantification Assignment

This assignment uses [Salmon](https://combine-lab.github.io/salmon/) to process RNA-seq data in FASTQ file format for three organisms: *Arabidopsis thaliana* (athal), human, and mouse.

## Assignment Overview

Complete the following tasks and submit your work as a "technical report" including the results. The codes/scripts used should be submitted as a separate file.

1. **[60 pts]** Process the RNA-seq FASTQ files of 'athal', 'human' and 'mouse'. Describe the steps taken for each processing in detail including the exact commands used for the tasks and the folder where the outputs are saved. Each case (athal, human and mouse) completed is 20 points each.

2. **[40 pts]** The main 'salmon' output is stored in 'quant.sf' file in a tab delimited format (Name, Length, EffectiveLength, TPM and NumReads). Compute TPM values using 1) Length and NumReads and 2) EffectiveLength and NumReads. Then, compare the computed TPM values and TPM (salmon output). Present the comparison results using scatter plot and correlation. Do this for all three cases (athal, human and mouse). Which one of Length or EffectiveLength is used by salmon to compute its TPM values?

## Available Data

The workspace contains:
- **FASTQ files**: `fastq/` directory with subdirectories for each organism
  - `athal/DRR016125/` - Arabidopsis thaliana RNA-seq data
  - `human/SRR8112669/` - Human RNA-seq data  
  - `mouse/SRR27016615/` - Mouse RNA-seq data
- **Reference transcriptomes**: `gene_annots/` directory with organism-specific annotations and transcriptome files

## Step-by-Step Guide

### Prerequisites

1. **Install Salmon** (if not already installed):
   ```bash
   # Using conda (recommended)
   conda config --add channels conda-forge
   conda config --add channels bioconda
   conda create -n salmon salmon
   conda activate salmon
   
   # Or download pre-compiled binaries from:
   # https://github.com/COMBINE-lab/salmon/releases
   ```

2. **Verify Salmon installation**:
   ```bash
   salmon -h
   ```

### Task 1: Process RNA-seq Data for All Three Organisms [60 pts]

#### A. Arabidopsis thaliana (athal) Processing [20 pts]

1. **Create output directory**:
   ```bash
   mkdir -p results/athal
   ```

2. **Build Salmon index**:
   ```bash
   # Using the available transcriptome file
   salmon index -t gene_annots/athal/athal.fa.gz -i results/athal/athal_index
   ```

3. **Quantify the sample**:
   ```bash
   salmon quant -i results/athal/athal_index -l A \
                -1 fastq/athal/DRR016125/DRR016125_1.fastq.gz \
                -2 fastq/athal/DRR016125/DRR016125_2.fastq.gz \
                -p 8 --validateMappings \
                -o results/athal/DRR016125_quant
   ```

4. **Output location**: `results/athal/DRR016125_quant/quant.sf`

#### B. Human Processing [20 pts]

1. **Create output directory**:
   ```bash
   mkdir -p results/human
   ```

2. **Build Salmon index** (using protein-coding transcripts):
   ```bash
   salmon index -t gene_annots/gencode/human/v46/gencode.v46.pc_transcripts.fa.gz \
                -i results/human/human_index
   ```

3. **Quantify the sample**:
   ```bash
   salmon quant -i results/human/human_index -l A \
                -1 fastq/human/SRR8112669/SRR8112669_1.fastq.gz \
                -2 fastq/human/SRR8112669/SRR8112669_2.fastq.gz \
                -p 8 --validateMappings \
                -o results/human/SRR8112669_quant
   ```

4. **Output location**: `results/human/SRR8112669_quant/quant.sf`

#### C. Mouse Processing [20 pts]

1. **Create output directory**:
   ```bash
   mkdir -p results/mouse
   ```

2. **Build Salmon index** (using protein-coding transcripts):
   ```bash
   salmon index -t gene_annots/gencode/mouse/M35_GRCm39/gencode.vM35.pc_transcripts.fa.gz \
                -i results/mouse/mouse_index
   ```

3. **Quantify the sample**:
   ```bash
   salmon quant -i results/mouse/mouse_index -l A \
                -1 fastq/mouse/SRR27016615/SRR27016615_1.fastq.gz \
                -2 fastq/mouse/SRR27016615/SRR27016615_2.fastq.gz \
                -p 8 --validateMappings \
                -o results/mouse/SRR27016615_quant
   ```

4. **Output location**: `results/mouse/SRR27016615_quant/quant.sf`

### Task 2: TPM Analysis and Comparison [40 pts]

The `quant.sf` file contains the following columns:
- **Name**: Transcript identifier
- **Length**: Transcript length
- **EffectiveLength**: Effective transcript length (accounting for fragment length distribution)
- **TPM**: Transcripts Per Million (Salmon's calculated value)
- **NumReads**: Estimated number of reads from this transcript

#### TPM Calculation Formula

TPM is calculated as:
```
TPM = (NumReads / Length) * (1,000,000 / sum(NumReads / Length for all transcripts))
```

#### Analysis Steps for Each Organism

1. **Load and examine the quant.sf file**:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.stats import pearsonr
   
   # Load data
   data = pd.read_csv('results/[organism]/[sample]_quant/quant.sf', sep='\t')
   ```

2. **Calculate TPM using Length**:
   ```python
   # Calculate TPM using Length
   reads_per_base_length = data['NumReads'] / data['Length']
   scaling_factor_length = 1e6 / reads_per_base_length.sum()
   tpm_calculated_length = reads_per_base_length * scaling_factor_length
   ```

3. **Calculate TPM using EffectiveLength**:
   ```python
   # Calculate TPM using EffectiveLength
   reads_per_base_eff = data['NumReads'] / data['EffectiveLength']
   scaling_factor_eff = 1e6 / reads_per_base_eff.sum()
   tpm_calculated_eff = reads_per_base_eff * scaling_factor_eff
   ```

4. **Compare with Salmon's TPM**:
   ```python
   # Calculate correlations
   corr_length = pearsonr(data['TPM'], tpm_calculated_length)[0]
   corr_eff = pearsonr(data['TPM'], tpm_calculated_eff)[0]
   
   # Create scatter plots
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Length-based TPM comparison
   ax1.scatter(data['TPM'], tpm_calculated_length, alpha=0.6)
   ax1.plot([0, data['TPM'].max()], [0, data['TPM'].max()], 'r--')
   ax1.set_xlabel('Salmon TPM')
   ax1.set_ylabel('Calculated TPM (Length)')
   ax1.set_title(f'Length-based (r={corr_length:.4f})')
   
   # EffectiveLength-based TPM comparison
   ax2.scatter(data['TPM'], tpm_calculated_eff, alpha=0.6)
   ax2.plot([0, data['TPM'].max()], [0, data['TPM'].max()], 'r--')
   ax2.set_xlabel('Salmon TPM')
   ax2.set_ylabel('Calculated TPM (EffectiveLength)')
   ax2.set_title(f'EffectiveLength-based (r={corr_eff:.4f})')
   
   plt.tight_layout()
   plt.savefig('tpm_comparison_[organism].png', dpi=300)
   plt.show()
   ```

#### Expected Results

The correlation analysis will reveal that **EffectiveLength** is used by Salmon to compute TPM values, as it accounts for:
- Fragment length distribution
- Sequence-specific biases
- Positional biases in RNA-seq data

The EffectiveLength-based calculation should show a correlation close to 1.0 with Salmon's TPM values, while the Length-based calculation will show a lower correlation.

## Key Salmon Parameters Explained

- **`-i`**: Path to the salmon index
- **`-l A`**: Auto-detect library type (strand orientation)
- **`-1`, `-2`**: Paths to paired-end read files
- **`-p`**: Number of threads to use
- **`--validateMappings`**: Enable mapping validation for improved accuracy
- **`-o`**: Output directory

## Output Files

Each quantification generates several files in the output directory:
- **`quant.sf`**: Main quantification results (transcript-level abundances)
- **`quant.genes.sf`**: Gene-level abundances (if gene mapping is available)
- **`aux_info/`**: Auxiliary information about the quantification
- **`logs/`**: Salmon log files
- **`libParams/`**: Library parameter estimates

## Notes

- Pre-built indices are available in the workspace for human and mouse (in the `gene_annots/` directories)
- Use protein-coding transcripts for better comparison across organisms
- The `--validateMappings` flag improves accuracy but increases runtime
- TPM values allow for comparison across samples and experiments