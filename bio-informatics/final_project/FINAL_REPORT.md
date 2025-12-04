# Integrative Analysis of Gene Expression, Clustering, and Differential Expression in Bladder Cancer Subtypes

**Authors:** Olawole Frankfurt Ogunfunminiyi, Niraj Kc  
**Affiliation:** Department of Computer Science and Electrical Engineering, West Virginia University  
**Date:** December 2024

---

## Abstract

**Motivation:** Bladder cancer (BLCA) exhibits significant molecular heterogeneity between low-grade and high-grade tumors. Understanding the transcriptomic differences between these subtypes is crucial for identifying potential biomarkers and therapeutic targets.

**Results:** We performed comprehensive transcriptomic analysis on TCGA-BLCA dataset comprising 90 samples (50 low-grade, 40 high-grade). After filtering lowly expressed genes (removing genes not expressed in ≥10% samples), 15,967 genes were retained for analysis. Unsupervised clustering identified 2 optimal clusters with entropy of 0.3902, indicating good separation between tumor grades. DESeq2 differential expression analysis revealed 2,146 significantly dysregulated genes (FDR < 0.01, |logFC| > 1), with 1,346 downregulated and 800 upregulated in high-grade tumors. Gene ontology enrichment analysis identified 153 pathways related to immune response, cell proliferation, and tissue development.

**Availability:** Data available from TCGA-BLCA project. Analysis code available upon request.

**Keywords:** Bladder cancer, TCGA-BLCA, Differential expression analysis, DESeq2, Clustering, Gene expression, Transcriptomics

---

## 1. Introduction

Bladder cancer (BLCA) is among the most common malignancies worldwide, with significant molecular heterogeneity that impacts prognosis and treatment response. The distinction between low-grade and high-grade bladder tumors is clinically critical, as high-grade tumors exhibit more aggressive behavior, higher recurrence rates, and poorer patient outcomes. Understanding the molecular mechanisms underlying these differences is essential for developing targeted therapeutic strategies and improving patient stratification.

The Cancer Genome Atlas (TCGA) project has generated comprehensive multi-omic datasets, including RNA-seq data, that enable systematic investigation of gene expression differences across cancer subtypes. In this study, we perform integrative transcriptomic analysis of TCGA-BLCA samples to: (1) characterize the global gene expression landscape through dimensionality reduction and unsupervised clustering, (2) identify differentially expressed genes (DEGs) between low-grade and high-grade tumors using rigorous statistical methods, and (3) interpret the biological significance of DEGs through gene ontology enrichment analysis.

Our analysis employs established bioinformatics methods including CPM normalization for gene expression, PCA for dimensionality reduction, K-means clustering with silhouette analysis for optimal cluster identification, DESeq2 for differential expression with proper RNA-seq count modeling, and gene set enrichment analysis for pathway interpretation. This comprehensive approach provides insights into the molecular distinctions between bladder cancer subtypes and identifies potential biomarkers and therapeutic targets.

---

## 2. Methods

### 2.1 Data Acquisition and Preprocessing

Gene expression count data and clinical annotations were obtained from the TCGA-BLCA project. The dataset comprised 90 tumor samples (50 low-grade, 40 high-grade) with RNA-seq read counts for protein-coding genes. Sample tumor grades were obtained from the class annotation file.

#### 2.1.1 Gene Filtering

To remove lowly expressed genes that contribute noise to downstream analyses, we applied the assignment-specified filtering strategy: "Filter out genes that are not expressed (count ≤ 5) in at least 10% of the samples". This means genes where count ≤ 5 in ≥10% of samples (≥9 samples) were removed. Equivalently, only genes with count ≤ 5 in <10% of samples were retained. This strict filtering approach ensures that genes with minimal expression across the cohort are excluded while preserving biologically relevant genes. After filtering, 15,967 genes were retained for analysis from an initial set of over 60,000 transcripts.

#### 2.1.2 Normalization

Count data were normalized using Counts Per Million (CPM) to account for differences in library sizes across samples:

```
CPM_ij = (count_ij / Σ_k count_kj) × 10^6
```

where count_ij is the raw count for gene i in sample j. For visualization and clustering analyses, we applied log transformation with a prior count of 2:

```
log2CPM_ij = log2(CPM_ij + 2)
```

### 2.2 Dimensionality Reduction and Clustering

#### 2.2.1 Principal Component Analysis

PCA was performed on standardized log₂CPM values to reduce dimensionality and visualize global expression patterns. Samples were standardized using z-score normalization prior to PCA. The first 10 principal components were computed, and variance explained by each component was evaluated.

#### 2.2.2 Optimal Cluster Determination

To identify natural groupings in the data, we performed K-means clustering with k ranging from 2 to 10. For each k, the silhouette score was calculated:

```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

where a(i) is the mean intra-cluster distance and b(i) is the mean nearest-cluster distance for sample i. The optimal k was selected as the value maximizing the mean silhouette score. We also performed hierarchical clustering with average linkage for comparison.

#### 2.2.3 Cluster Quality Assessment

Cluster quality was assessed using entropy, which measures the agreement between predicted clusters and true tumor grades. For each cluster j, we computed:

```
e_j = -Σ_i p_ij log2(p_ij)
```

where p_ij is the proportion of samples in cluster j belonging to true class i. Total entropy was calculated as the weighted average:

```
E = Σ_j (m_j/m) × e_j
```

where m_j is the size of cluster j and m is the total number of samples. Lower entropy indicates better alignment between clusters and true labels.

### 2.3 Differential Expression Analysis

#### 2.3.1 DESeq2 Analysis

Differential expression analysis was performed using DESeq2, which models RNA-seq count data using a negative binomial distribution. DESeq2 accounts for library size differences, estimates gene-wise dispersion, and applies shrinkage to improve stability of log fold change estimates. The statistical model is:

```
K_ij ~ NB(μ_ij, α_i)
```

where K_ij is the count for gene i in sample j, μ_ij is the mean, and α_i is the dispersion parameter. Wald tests were used to assess significance, with p-values adjusted for multiple testing using the Benjamini-Hochberg procedure.

Genes were considered significantly differentially expressed if they met the criteria: adjusted p-value (FDR) < 0.01 and absolute log₂ fold change > 1.

### 2.4 Gene Ontology Enrichment Analysis

Gene ontology (GO) enrichment analysis was performed using the Enrichr tool via GSEApy. Significantly differentially expressed genes were tested for enrichment in GO Biological Process, Molecular Function, and Cellular Component gene sets (2023 versions). Enriched terms with adjusted p-value < 0.05 were considered significant.

---

## 3. Results

### 3.1 Dataset Characteristics and Gene Filtering

The TCGA-BLCA dataset comprised 90 tumor samples with known tumor grades: 50 low-grade and 40 high-grade samples. Initial expression data contained counts for 60,660 genomic features. After applying the assignment-specified filtering ("filter out genes that are not expressed (count ≤ 5) in at least 10% of samples"), 15,967 genes were retained for downstream analysis. This filtering removed 44,693 lowly expressed genes (73.7% of original features), reducing noise while preserving biologically relevant transcripts.

### 3.2 Global Gene Expression Patterns

Principal component analysis of log₂CPM-normalized expression revealed the dominant axes of variation in the dataset. PC1 and PC2 explained 13.59% and 9.18% of total variance, respectively, for a cumulative variance of 22.77%. The PCA scatter plot showed partial separation between low-grade and high-grade samples, with considerable overlap indicating molecular heterogeneity within tumor grades. The scree plot demonstrated that variance was distributed across multiple components rather than concentrated in the first few PCs, suggesting complex underlying biological processes.

![PCA Analysis](results/task1_pca_plot.png)
**Figure 1:** PCA analysis of TCGA-BLCA samples. Left: PCA scatter plot showing partial separation of Low Grade (blue) and High Grade (red) tumors along PC1 and PC2. Right: Scree plot showing variance explained by the first 10 principal components.

### 3.3 Unsupervised Clustering Analysis

#### 3.3.1 Optimal Cluster Identification

Silhouette analysis was performed for K-means and hierarchical clustering with k ranging from 2 to 10. For K-means clustering, silhouette scores ranged from 0.0721 (k=2) to 0.0412 (k=9), with the maximum score of 0.0721 at k=2. For hierarchical clustering, silhouette scores ranged from 0.4544 (k=2) to 0.0500 (k=10), achieving maximum at k=2. Both methods consistently identified k=2 as optimal, with hierarchical clustering showing stronger separation, suggesting two natural groupings in the expression data.

![Silhouette Analysis](results/task2_silhouette_analysis.png)
**Figure 2:** Silhouette analysis for optimal cluster selection. Left: K-means clustering shows optimal k=2 with score 0.0721. Right: Hierarchical clustering shows optimal k=2 with score 0.4544, indicating stronger cluster separation.

#### 3.3.2 Cluster Composition and Quality

Using K-means with k=2, Cluster 0 contained 53 samples (5 high-grade, 48 low-grade) while Cluster 1 contained 37 samples (35 high-grade, 2 low-grade). The cluster distribution shows good enrichment for each tumor grade, with Cluster 0 being 90.6% low-grade and Cluster 1 being 94.6% high-grade.

Entropy analysis quantified the alignment between cluster assignments and true tumor grades. Cluster 0 had entropy e₀ = 0.4508, and Cluster 1 had entropy e₁ = 0.3034. The total weighted entropy was E = 0.3902, normalized as 0.3902 relative to maximum possible entropy per cluster (log₂(2) = 1.0). This moderate entropy indicates reasonable separation of tumor grades by unsupervised clustering, demonstrating that molecular signatures can partially distinguish the two grades without supervision.

![Cluster Visualization](results/task2_cluster_visualization.png)
**Figure 3:** Cluster visualization in PCA space. Left: K-means clustering with k=2 showing cluster assignments. Right: True tumor grade labels, demonstrating good agreement between unsupervised clustering and clinical grades.

![Confusion Matrix](results/task2_confusion_matrix.png)
**Figure 4:** Confusion matrix comparing cluster assignments to true tumor grades. Cluster 0 predominantly contains Low Grade tumors (48/53), while Cluster 1 predominantly contains High Grade tumors (35/37).

![Heatmap](results/task2_heatmap.png)
**Figure 5:** Heatmap of top 50 most variable genes showing hierarchical clustering of samples. Column colors indicate cluster assignments, revealing distinct expression patterns between sample groups.

### 3.4 Differential Expression Analysis

#### 3.4.1 DESeq2 Identifies Extensive Transcriptional Dysregulation

DESeq2 analysis comparing high-grade versus low-grade tumors identified 2,146 significantly differentially expressed genes (FDR < 0.01, |log₂FC| > 1) out of 15,967 genes tested (13.4%). Of these, 1,346 genes (62.7%) were downregulated and 800 genes (37.3%) were upregulated in high-grade tumors relative to low-grade tumors. The magnitude of dysregulation ranged from log₂FC = -9.54 to +9.54, indicating dramatic expression changes for certain genes.

Top upregulated genes in high-grade tumors included ENSG00000231683 (log₂FC = 9.54, FDR = 2.8e-15), ENSG00000185479 (log₂FC = 9.45, FDR = 3.8e-67), and ENSG00000170454 (log₂FC = 8.58, FDR = 4.7e-36). Top downregulated genes included ENSG00000260676 (log₂FC = -9.54, FDR = 3.2e-12), ENSG00000166863 (log₂FC = -9.35, FDR = 1.8e-58), and ENSG00000162877 (log₂FC = -8.69, FDR = 1.7e-66).

![Volcano Plot](results/task3_volcano_plot.png)
**Figure 6:** Volcano plot of differential expression analysis. Red points indicate upregulated genes (800), blue points indicate downregulated genes (1,346), and gray points are non-significant. Significance thresholds: FDR < 0.01 and |log₂FC| > 1.

#### 3.4.2 DEG-Based Dimensionality Reduction

PCA performed on the 2,146 significant DEGs revealed improved separation compared to all-gene PCA. PC1 and PC2 explained 28.72% and 9.04% of variance, respectively (cumulative 37.77%). This substantial increase in variance explained (37.77% vs 22.77%) demonstrates that DEGs capture the primary sources of variation distinguishing tumor grades.

![DEG PCA](results/task3_pca_degs.png)
**Figure 7:** PCA analysis using only the 2,146 significant DEGs. Left: Clear separation of tumor grades along PC1 (28.72% variance). Right: Scree plot showing improved variance capture compared to all-gene PCA.

### 3.5 Biological Interpretation via Gene Ontology

Gene ontology enrichment analysis of all 2,146 DEGs identified 153 significantly enriched Biological Process terms (FDR < 0.05). Top enriched processes included:

- **Immune response and inflammation:** cytokine-mediated signaling pathway (FDR = 1.2e-15, 89 genes), inflammatory response (FDR = 3.4e-12, 67 genes), leukocyte migration (FDR = 8.9e-11, 54 genes)
- **Cell proliferation and division:** mitotic cell cycle (FDR = 2.1e-08, 78 genes), cell division (FDR = 5.6e-07, 65 genes), chromosome segregation (FDR = 1.3e-06, 42 genes)
- **Tissue development and morphogenesis:** epithelial cell differentiation (FDR = 4.5e-09, 71 genes), tissue development (FDR = 7.8e-08, 93 genes), extracellular matrix organization (FDR = 2.3e-07, 58 genes)

These enriched pathways implicate dysregulation of immune surveillance, uncontrolled proliferation, and disrupted tissue architecture as key molecular features distinguishing high-grade from low-grade bladder tumors.

![GO Enrichment](results/task4_go_enrichment_plots.png)
**Figure 8:** Gene Ontology enrichment analysis. Top: Bar plot of top 20 enriched GO Biological Process terms ranked by -log₁₀(adjusted p-value). Bottom: Dot plot showing relationship between gene count, significance, and adjusted p-value for enriched terms.

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

Our comprehensive transcriptomic analysis of TCGA-BLCA samples reveals extensive molecular differences between low-grade and high-grade bladder tumors. The identification of 2,146 differentially expressed genes (13.4% of tested genes) underscores the substantial transcriptional reprogramming accompanying bladder cancer progression.

### 4.1 Methodological Considerations

The use of count-based gene filtering rather than CPM-based filtering is critical for RNA-seq analysis, as it prevents bias toward highly expressed genes and ensures that lowly expressed genes present in a subset of samples are appropriately excluded based on detection frequency rather than relative abundance. Our threshold of raw count > 5 in at least 10% of samples is consistent with established best practices for bulk RNA-seq preprocessing.

The application of DESeq2 for differential expression analysis is methodologically superior to simpler approaches like t-tests on normalized counts. DESeq2's negative binomial model explicitly accounts for the discrete nature of count data, overdispersion, and library size differences. The shrinkage of log fold change estimates improves robustness, particularly for genes with low counts or high variability. Our results demonstrate the power of proper statistical modeling, with thousands of DEGs identified at stringent significance thresholds.

### 4.2 Biological Insights

The moderate entropy (0.3902) observed in unsupervised clustering indicates that global expression patterns can reasonably separate tumor grades, with Cluster 0 enriched for low-grade (90.6%) and Cluster 1 enriched for high-grade (94.6%) tumors. This finding demonstrates that grade-specific molecular signatures exist and can be detected through unsupervised methods. The improved PCA separation using DEGs alone (37.77% variance explained vs 22.77% for all genes) further confirms that differential expression is concentrated in specific biological pathways rather than representing random transcriptional noise.

Gene ontology enrichment analysis revealed dysregulation of immune response pathways, consistent with known roles of immune evasion and tumor microenvironment remodeling in cancer progression. The enrichment of cell proliferation and cell cycle pathways aligns with the increased mitotic activity characteristic of high-grade tumors. Alterations in tissue development and extracellular matrix organization suggest disruption of normal epithelial architecture, a hallmark of invasive cancer.

### 4.3 Limitations and Future Directions

This analysis focused on protein-coding genes and did not examine non-coding RNAs, which may also contribute to tumor grade differences. Additionally, bulk RNA-seq data represent averaged signals across heterogeneous cell populations; single-cell RNA-seq would provide finer resolution of cellular subpopulations. Future work should validate key DEGs at the protein level and investigate their functional roles in bladder cancer progression through experimental perturbation studies.

---

## 5. Conclusion

This study presents a comprehensive transcriptomic analysis of bladder cancer subtypes using TCGA-BLCA data. Through rigorous preprocessing following assignment specifications, statistical analysis with DESeq2, and biological interpretation, we identified 2,146 significantly differentially expressed genes between low-grade and high-grade tumors. Gene ontology enrichment revealed dysregulation of immune response, cell proliferation, and tissue architecture pathways (153 enriched terms) as molecular hallmarks of high-grade disease. These findings contribute to our understanding of bladder cancer molecular heterogeneity and provide candidate biomarkers for future validation studies.

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
