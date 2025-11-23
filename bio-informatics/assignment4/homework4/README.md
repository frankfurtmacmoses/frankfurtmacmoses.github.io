This assignment is to utilize students' knowledge in identifying differentially expressed genes from RNA sequencing data and visualizing the results.   Download the data (RNAseq count data Download RNAseq count dataand class labels Download class labels).  'RNAseq count data' are comma-separated text file, containing 13 columns where the first four columns are gene annotations (gene_id, gene_type, gene_name, hgnc_id) and the rest are RNAseq count data for 9 samples.  'class labels' has the class labels (TP: tumor primary, NT: non-tumor) for those sample.  

To perform the tasks below, use either edgeRLinks to an external site. or DESeq2Links to an external site..  You will find the following tutorials for edgeRLinks to an external site. (anotherLinks to an external site.) and DESeq2Links to an external site. (anotherLinks to an external site.) useful, respectively.

[20 pts] Perform PCA using "normalized counts" which can be obtained via 'cpm' (edgeR) or 'counts(., normalized = TRUE)' (DESeq2) function.  Each sample should be color-coded by class (TP vs. NP) to visualization of class distributions. 
[40 pts] Perform differential expression analysis to identify differentially expressed genes between two classes: TP vs. NT.  Consider the following steps and filtering:
Remove low-count genes (genes have a count of at least 5 for a 3 samples, before DEG analysis.
Use adjusted p-value (FDR) < 0.01 to identify differentially expressed genes.
Save the results in a text file (comma or tab separated file) for submission.
[20 pts] Perform PCA using "normalized counts" of DEGs.  Compare the PCA plots before DEG analysis and after.  
[20 pts] Visualize RNAseq normalized counts of DEGs via heatmap.  There are various R packages to support creating heatmap: ComplexHeatmapLinks to an external site., pheatmapLinks to an external site., or default heatmap()Links to an external site. function. 

dataset has been renamed