# Integrative Analysis of Gene Expression, Clustering, and Differential Expression in Bladder Cancer Subtypes

**Authors:** Olawole Frankfurt Ogunfunminiyi, Niraj Kc  
**Affiliation:** Department of Computer Science and Electrical Engineering, West Virginia University  
**Date:** December 2024

---

## Abstract

**Motivation:** Bladder cancer shows considerable variation between low-grade and high-grade tumors at the molecular level. We sought to understand these transcriptomic differences to better identify biomarkers and potential therapeutic targets.

**Results:** Our transcriptomic analysis of the TCGA-BLCA dataset included 90 tumor samples—50 low-grade and 40 high-grade. We filtered out genes with minimal expression (those not expressed in at least 10% of samples), leaving 15,967 genes for downstream analysis. Through unsupervised clustering, we found 2 optimal clusters with an entropy of 0.3902, suggesting reasonable separation between tumor grades. Using DESeq2 for differential expression, we identified 2,146 genes showing significant changes (FDR < 0.01, |logFC| > 1), with 1,346 genes downregulated and 800 upregulated in high-grade compared to low-grade tumors. Gene ontology enrichment revealed 153 biological process pathways involved in immune response, cell proliferation, and tissue development.

**Availability:** Data from the TCGA-BLCA project. Analysis scripts available upon request.

**Keywords:** Bladder cancer, TCGA-BLCA, Differential expression, DESeq2, Clustering, Transcriptomics

---

## 1. Introduction

Bladder cancer ranks among the most common malignancies globally and shows substantial molecular diversity that affects both prognosis and how patients respond to treatment. Distinguishing between low-grade and high-grade bladder tumors matters clinically because high-grade tumors behave more aggressively, recur more frequently, and result in worse outcomes for patients. Understanding what drives these differences at the molecular level could help us develop better targeted therapies and improve how we classify patients for treatment.

The Cancer Genome Atlas (TCGA) project has provided extensive multi-omic datasets, including RNA sequencing data, which allow us to systematically examine gene expression patterns across different cancer types. For this study, we analyzed TCGA-BLCA samples with three main goals: first, to characterize the overall gene expression patterns using dimensionality reduction and unsupervised clustering; second, to identify genes that are differentially expressed between low-grade and high-grade tumors using robust statistical approaches; and third, to understand the biological meaning of these gene expression changes through pathway enrichment analysis.

We used several standard bioinformatics approaches: CPM normalization to adjust for differences in sequencing depth, principal component analysis (PCA) to reduce data complexity, K-means clustering combined with silhouette analysis to find natural groupings in the data, DESeq2 to properly model count-based differential expression, and gene ontology enrichment to interpret biological pathways. Together, these methods help reveal the molecular features that distinguish bladder cancer subtypes and point to potential biomarkers and therapeutic targets.

---

## 2. Methods

### 2.1 Data Acquisition and Preprocessing

We obtained gene expression count data and clinical information from the TCGA-BLCA project. The dataset includes 90 tumor samples—50 low-grade and 40 high-grade—with RNA-seq read counts for protein-coding genes. Tumor grade information came from the class annotation file.

#### 2.1.1 Gene Filtering

Lowly expressed genes can add noise to the analysis, so we filtered them out following the guideline: "Filter out genes that are not expressed (count ≤ 5) in at least 10% of the samples." In practical terms, we removed genes where the count was 5 or less in 9 or more samples (that's 10% of our 90 samples). Put another way, we only kept genes that had counts above 5 in at least 90% of samples. This stringent filtering removes genes with minimal expression while keeping those that are biologically relevant. Starting with over 60,000 transcripts, we ended up with 15,967 genes for our analysis.

#### 2.1.2 Normalization

To account for differences in sequencing depth between samples, we normalized the count data using Counts Per Million (CPM):

```
CPM_ij = (count_ij / Σ_k count_kj) × 10^6
```

where count_ij represents the raw count for gene i in sample j. For visualization and clustering, we log-transformed the CPM values using a prior count of 2:

```
log2CPM_ij = log2(CPM_ij + 2)
```

This transformation helps stabilize variance and makes the data more suitable for downstream analyses.

### 2.2 Dimensionality Reduction and Clustering

#### 2.2.1 Principal Component Analysis

We applied PCA to the standardized log₂CPM values to reduce the dimensionality of our data and visualize overall expression patterns. Before running PCA, we standardized samples using z-score normalization. We computed the first 10 principal components and looked at how much variance each one explained.

#### 2.2.2 Optimal Cluster Determination

To find natural groupings in the expression data, we ran K-means clustering trying k values from 2 to 10. For each k, we calculated the silhouette score:

```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

where a(i) is how far a sample is from others in its cluster on average, and b(i) is how far it is from samples in the nearest other cluster. We picked the k that gave the highest average silhouette score. We also tried hierarchical clustering with average linkage to see how the results compared.

#### 2.2.3 Cluster Quality Assessment

We assessed how well our clusters matched the actual tumor grades using entropy, which measures agreement between cluster assignments and true labels. For each cluster j, we calculated:

```
e_j = -Σ_i p_ij log2(p_ij)
```

where p_ij is the fraction of samples in cluster j that actually belong to grade i. Total entropy is the weighted average across all clusters:

```
E = Σ_j (m_j/m) × e_j
```

where m_j is the number of samples in cluster j and m is the total sample count. Lower entropy means better agreement between our clusters and the real tumor grades.

### 2.3 Differential Expression Analysis

#### 2.3.1 DESeq2 Analysis

For differential expression, we used DESeq2, which models RNA-seq count data with a negative binomial distribution. DESeq2 handles several important issues: it accounts for differences in library sizes, estimates how much each gene varies, and shrinks fold change estimates to make them more reliable. The statistical model looks like this:

```
K_ij ~ NB(μ_ij, α_i)
```

where K_ij is the count for gene i in sample j, μ_ij is the expected value, and α_i captures the dispersion. We used Wald tests to check significance and adjusted p-values for multiple testing using the Benjamini-Hochberg method.

We considered a gene significantly differentially expressed if it had an adjusted p-value (FDR) below 0.01 and an absolute log₂ fold change greater than 1.

### 2.4 Gene Ontology Enrichment Analysis

Gene ontology (GO) enrichment analysis was performed using the Enrichr tool via GSEApy. Significantly differentially expressed genes were tested for enrichment in GO Biological Process, Molecular Function, and Cellular Component gene sets (2023 versions). Enriched terms with adjusted p-value < 0.05 were considered significant.

---

## 3. Results

### 3.1 Dataset Characteristics and Gene Filtering

The TCGA-BLCA dataset comprised 90 tumor samples with known tumor grades: 50 low-grade and 40 high-grade samples. Initial expression data contained counts for 60,660 genomic features. After applying the assignment-specified filtering ("filter out genes that are not expressed (count ≤ 5) in at least 10% of samples"), 15,967 genes were retained for downstream analysis. This filtering removed 44,693 lowly expressed genes (73.7% of original features), reducing noise while preserving biologically relevant transcripts.

### 3.2 Global Gene Expression Patterns

When we ran PCA on the log₂CPM-normalized expression data, we found that PC1 and PC2 captured 13.59% and 9.18% of the total variance, respectively—adding up to 22.77%. Looking at the PCA scatter plot, low-grade and high-grade samples showed some separation, though there was still considerable overlap. This suggests molecular heterogeneity exists even within each tumor grade. The scree plot showed that variance spread across multiple components rather than being concentrated in just the first few, which makes sense given the complexity of cancer biology.

![PCA Analysis](results/task1_pca_plot.png)
**Figure 1:** PCA analysis of TCGA-BLCA samples. Left: PCA scatter plot showing partial separation of Low Grade (blue) and High Grade (red) tumors along PC1 and PC2. Right: Scree plot showing variance explained by the first 10 principal components.

### 3.3 Unsupervised Clustering Analysis

#### 3.3.1 Optimal Cluster Identification

We tested both K-means and hierarchical clustering with k ranging from 2 to 10 clusters. For K-means, silhouette scores went from 0.0721 at k=2 down to 0.0412 at k=9, with the best score at k=2. Hierarchical clustering performed better overall, with scores ranging from 0.4544 at k=2 down to 0.0500 at k=10, also peaking at k=2. Both methods pointed to two natural groups in the data, though hierarchical clustering showed clearer separation between them.

![Silhouette Analysis](results/task2_silhouette_analysis.png)
**Figure 2:** Silhouette analysis for optimal cluster selection. Left: K-means clustering shows optimal k=2 with score 0.0721. Right: Hierarchical clustering shows optimal k=2 with score 0.4544, indicating stronger cluster separation.

#### 3.3.2 Cluster Composition and Quality

When we used K-means with k=2, Cluster 0 ended up with 53 samples (48 low-grade and 5 high-grade), while Cluster 1 had 37 samples (35 high-grade and 2 low-grade). This represents pretty good enrichment—Cluster 0 was 90.6% low-grade and Cluster 1 was 94.6% high-grade.

We quantified how well the clusters matched actual tumor grades using entropy. Cluster 0 had an entropy of 0.4508, and Cluster 1 had an entropy of 0.3034. The overall weighted entropy came out to 0.3902, which is moderate (where 0 is perfect and 1 is random). This tells us that unsupervised clustering does a reasonable job separating tumor grades, showing that molecular signatures can distinguish between the two grades even without using the labels.

![Cluster Visualization](results/task2_cluster_visualization.png)
**Figure 3:** Cluster visualization in PCA space. Left: K-means clustering with k=2 showing cluster assignments. Right: True tumor grade labels, demonstrating good agreement between unsupervised clustering and clinical grades.

![Confusion Matrix](results/task2_confusion_matrix.png)
**Figure 4:** Confusion matrix comparing cluster assignments to true tumor grades. Cluster 0 predominantly contains Low Grade tumors (48/53), while Cluster 1 predominantly contains High Grade tumors (35/37).

![Heatmap](results/task2_heatmap.png)
**Figure 5:** Heatmap of top 50 most variable genes showing hierarchical clustering of samples. Column colors indicate cluster assignments, revealing distinct expression patterns between sample groups.

### 3.4 Differential Expression Analysis

#### 3.4.1 DESeq2 Identifies Extensive Transcriptional Dysregulation

When we compared high-grade to low-grade tumors using DESeq2, we found 2,146 genes with significant expression changes (FDR < 0.01, |log₂FC| > 1) out of 15,967 tested genes—that's about 13.4%. Most of these changes involved downregulation: 1,346 genes (62.7%) went down in high-grade tumors, while 800 genes (37.3%) went up. Some genes showed really dramatic shifts, with fold changes ranging from -9.54 to +9.54 on the log₂ scale.

The most upregulated genes in high-grade tumors were ENSG00000231683 (log₂FC = 9.54, FDR = 2.8e-15), ENSG00000185479 (log₂FC = 9.45, FDR = 3.8e-67), and ENSG00000170454 (log₂FC = 8.58, FDR = 4.7e-36). On the flip side, the most downregulated were ENSG00000260676 (log₂FC = -9.54, FDR = 3.2e-12), ENSG00000166863 (log₂FC = -9.35, FDR = 1.8e-58), and ENSG00000162877 (log₂FC = -8.69, FDR = 1.7e-66).

![Volcano Plot](results/task3_volcano_plot.png)
**Figure 6:** Volcano plot of differential expression analysis. Red points indicate upregulated genes (800), blue points indicate downregulated genes (1,346), and gray points are non-significant. Significance thresholds: FDR < 0.01 and |log₂FC| > 1.

#### 3.4.2 DEG-Based Dimensionality Reduction

When we ran PCA using only the 2,146 significant DEGs, we saw much better separation between tumor grades compared to using all genes. PC1 and PC2 now explained 28.72% and 9.04% of variance (37.77% total). This jump from 22.77% to 37.77% tells us that the DEGs really do capture the main differences between tumor grades.

![DEG PCA](results/task3_pca_degs.png)
**Figure 7:** PCA analysis using only the 2,146 significant DEGs. Left: Clear separation of tumor grades along PC1 (28.72% variance). Right: Scree plot showing improved variance capture compared to all-gene PCA.

### 3.5 Biological Interpretation via Gene Ontology

When we looked at what biological processes our 2,146 DEGs were involved in, gene ontology enrichment turned up 153 significantly enriched terms (FDR < 0.05). The top hits fell into a few main categories:

- **Immune response and inflammation:** cytokine signaling (FDR = 1.2e-15, 89 genes), inflammatory response (FDR = 3.4e-12, 67 genes), and leukocyte migration (FDR = 8.9e-11, 54 genes)
- **Cell proliferation and division:** mitotic cell cycle (FDR = 2.1e-08, 78 genes), cell division (FDR = 5.6e-07, 65 genes), and chromosome segregation (FDR = 1.3e-06, 42 genes)
- **Tissue development and structure:** epithelial cell differentiation (FDR = 4.5e-09, 71 genes), tissue development (FDR = 7.8e-08, 93 genes), and extracellular matrix organization (FDR = 2.3e-07, 58 genes)

These results point to disrupted immune surveillance, uncontrolled cell division, and altered tissue organization as major molecular differences between high-grade and low-grade bladder tumors.

![GO Enrichment](results/task4_go_enrichment_plots.png)
**Figure 8:** Gene Ontology enrichment analysis. Top: Bar plot of top 20 enriched GO Biological Process terms ranked by -log₁₀(adjusted p-value). Bottom: Dot plot showing relationship between gene count, significance, and adjusted p-value for enriched terms.

![GO Enrichment Labeled](results/task4_go_enrichment_labeled.png)
**Figure 9:** Enhanced GO enrichment dot plot with key terms labeled. Bubble size shows gene count, with highlighted annotations for major processes: Cell Population Proliferation, Cell Migration, Inflammatory Response, and Angiogenesis.

### 3.6 Summary Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 90 |
| Low Grade Samples | 50 |
| High Grade Samples | 40 |
| Initial Gene Features | 60,660 |
| Genes After Filtering | 15,967 |
| Optimal Clusters (K-means) | 2 |
| Silhouette Score K-means (k=2) | 0.0721 |
| Silhouette Score Hierarchical (k=2) | 0.4544 |
| Cluster Entropy | 0.3902 |
| Significant DEGs (FDR<0.01, \|logFC\|>1) | 2,146 |
| Upregulated in High Grade | 800 |
| Downregulated in High Grade | 1,346 |
| PC1+PC2 Variance (All Genes) | 22.77% |
| PC1+PC2 Variance (DEGs Only) | 37.77% |
| Enriched GO:BP Terms (FDR<0.05) | 153 |

---

## 4. Discussion

Our analysis of TCGA-BLCA samples uncovered substantial molecular differences between low-grade and high-grade bladder tumors. Finding 2,146 differentially expressed genes—about 13.4% of all genes we tested—shows just how much transcriptional rewiring happens as bladder cancer progresses.

### 4.1 Methodological Considerations

Using count-based filtering instead of CPM-based filtering was important here. Count-based filtering avoids favoring highly expressed genes and makes sure we're removing lowly expressed genes based on how often they're detected, not how abundant they are when present. Our cutoff (raw count > 5 in at least 10% of samples) follows standard practices for bulk RNA-seq data preprocessing.

Choosing DESeq2 for differential expression was also key. It's more appropriate than simpler methods like t-tests on normalized counts because it accounts for the fact that RNA-seq data are discrete counts, not continuous measurements. DESeq2 handles overdispersion and library size differences properly, and it shrinks fold change estimates to make them more reliable, especially for genes with low counts or high variability. The fact that we identified thousands of DEGs at strict significance cutoffs shows the value of using the right statistical approach.

### 4.2 Biological Insights

The moderate entropy (0.3902) from our unsupervised clustering suggests that overall expression patterns do a decent job separating tumor grades—Cluster 0 was 90.6% low-grade and Cluster 1 was 94.6% high-grade. This tells us grade-specific molecular signatures really do exist and can be picked up without using the grade labels. The fact that PCA worked better with just the DEGs (explaining 37.77% vs. 22.77% of variance) confirms that the expression differences aren't random noise but are concentrated in particular biological pathways.

The gene ontology results make biological sense. Immune response pathways showed up prominently, which fits with what we know about immune evasion and microenvironment changes in cancer. We also saw enrichment in cell proliferation and cell cycle pathways, matching the increased cell division in high-grade tumors. Changes in tissue development and extracellular matrix organization suggest the normal epithelial structure is breaking down—a key feature of invasive cancer.

### 4.3 Limitations and Future Directions

We focused on protein-coding genes here, so we didn't look at non-coding RNAs, which might also play a role in tumor grade differences. Bulk RNA-seq gives us averaged signals across mixed cell populations, so we can't see what's happening in specific cell types. Single-cell RNA-seq would give us that finer detail. Going forward, it would be valuable to validate some of these key DEGs at the protein level and test their actual functional roles in bladder cancer progression through lab experiments.

---

## 5. Conclusion

This study provides a thorough transcriptomic look at bladder cancer subtypes using data from TCGA-BLCA. Through careful data preprocessing, statistical analysis with DESeq2, and biological interpretation, we identified 2,146 genes with significant expression differences between low-grade and high-grade tumors. Gene ontology enrichment pointed to dysregulation in immune response, cell proliferation, and tissue organization pathways (153 enriched terms total) as key molecular features of high-grade disease. These results add to our understanding of molecular heterogeneity in bladder cancer and suggest potential biomarkers that could be validated in future studies.

---

## Competing Interests

No competing interest is declared.

---

## Author Contributions

F.M. conceived and designed the study, performed all data analyses, generated visualizations, and wrote the manuscript.

---

## Acknowledgments

Data were generated by the TCGA Research Network. The author acknowledges West Virginia University for computational resources and support.

---

## References

1. The Cancer Genome Atlas Research Network. Comprehensive molecular characterization of urothelial bladder carcinoma. *Nature*, 507(7492):315-322, 2014.

2. Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*, 15(12):550, 2014.

3. Robinson MD, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. *Bioinformatics*, 26(1):139-140, 2010.

4. Xie C, Mao X, Huang J, et al. KOBAS 2.0: a web server for annotation and identification of enriched pathways and diseases. *Nucleic Acids Research*, 41(W1):W71-W77, 2013.

5. Kuleshov MV, Jones MR, Rouillard AD, et al. Enrichr: a comprehensive gene set enrichment analysis web server 2016 update. *Nucleic Acids Research*, 44(W1):W90-W97, 2016.

6. Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20:53-65, 1987.

7. Shannon CE. A mathematical theory of communication. *The Bell System Technical Journal*, 27(3):379-423, 1948.

---

**Report Prepared By:** Frankfurt MacMoses  
**Date:** November 28, 2025  
**Institution:** West Virginia University
