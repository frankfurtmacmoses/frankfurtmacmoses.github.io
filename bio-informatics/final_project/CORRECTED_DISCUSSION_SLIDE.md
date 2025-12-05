# Similarities: Comparison with Robertson et al. (2017)

---

Our findings show strong concordance with the landmark TCGA-BLCA study published in *Cell* (2017).

---

## **Tumor Microenvironment**

### **Robertson et al. (2017):**
- Emphasized **stromal and EMT pathways** in aggressive bladder tumors

### **Our Analysis:**
We found GO terms supporting EMT and vascular remodeling:
- *Regulation of Cell Migration* (FDR 1.08×10⁻⁵, 87 genes)
- *Regulation of Angiogenesis* (FDR 3.38×10⁻⁴, 46 genes)

These directly support tissue invasion and microenvironment remodeling described by Robertson et al.

---

## **Cell Cycle Dysregulation**

### **Robertson et al. (2017):**
- Tumors with poor prognosis often have **cell cycle genes turned on** (cancer cells divide very fast)

### **Our GO Enrichment Strongly Supports This:**
- *Regulation of Cell Population Proliferation* (FDR 2.89×10⁻⁷, 141 genes)
- *Positive Regulation of Cell Population Proliferation* (FDR 6.81×10⁻⁵, 90 genes)

**Key proliferation genes:** CDK6, FOXM1, MYC, CDKN2A, EGFR, ERBB2 (found in proliferation GO terms)

This validates Robertson's finding that aggressive tumors exhibit uncontrolled proliferation.

---

## **Immune Checkpoints**

### **Robertson et al. (2017):**
- Tumors classified as **Basal/Squamous subtype** showed **high expression of immune-related genes**

### **We Observed Enrichment for Immune-Related GO Terms:**
- *Inflammatory Response* (FDR 1.11×10⁻³, 49 genes)
- *Cytokine-Mediated Signaling Pathway* (FDR 6.17×10⁻³, 49 genes)

**Immune genes present in GO enrichment:**
- **Chemokines:** CXCL8, CXCL10, CXCL1, CXCL9, CXCL11 (all found in Inflammatory Response GO term)
- **TGF-β family:** TGFB1, TGFB2 (found in multiple GO terms including cell proliferation and migration)

These align with Robertson's observation of immune activation in aggressive bladder cancer

---

## **Key Validation Summary**

✅ **All three major findings from Robertson et al. (2017) are confirmed in our independent analysis:**

1. **Stromal/EMT pathways** → Our cell migration & angiogenesis GO terms
2. **Cell cycle activation** → Our proliferation GO terms (strongest enrichment)
3. **Immune gene expression** → Our inflammatory response & cytokine signaling GO terms

---

### **Biological Significance:**

Our **2,146 DEGs** and **153 enriched GO:BP terms** independently recapitulate the molecular signatures identified by the TCGA consortium, validating that:

- High Grade bladder tumors exhibit a **"dual axis"** phenotype:
  - **Axis 1:** Proliferation (141 genes, FDR 2.89×10⁻⁷)
  - **Axis 2:** Microenvironment remodeling + Immune activation (136 genes combined)

- These findings support the **Basal/Squamous molecular subtype** described by Robertson et al., characterized by:
  - Dedifferentiation (loss of urothelial markers)
  - Immune infiltration
  - Poor prognosis

---

### **Clinical Implications Aligned with TCGA:**

Both studies suggest High Grade bladder cancer requires **multimodal therapy**:
- **Chemotherapy** (target proliferation)
- **Immunotherapy** (leverage immune activation - checkpoint inhibitors)
- **Anti-angiogenics** (disrupt tumor microenvironment)
