## 5. Experiments

In this section, we present the quantitative and qualitative results of our evaluation. We benchmark three state-of-the-art multi-subject personalization models: **MOSAIC**, **XVerse**, and **PSR**, across five difficulty levels ranging from 2 to 10 interacting subjects.

### 5.1. Quantitative Results: The Limits of Personalization

To rigorously assess model performance, we first examine the overall trend of identity preservation as the subject count increases. The quantitative results, averaged across all scene types, are summarized in **Table 1** and visualized in **Figure 2**.

*(Placeholder: Table 1. Insert a comprehensive table here. Columns should include: Model, Subject Count (2, 4, 6, 8, 10), CLIP-T, CLIP-I, DINOv2, and SCR@0.4. Populate the data directly from the *_summary.csv files in eval_outputs.)*

*(Placeholder: Figure 2. Insert two line charts side-by-side. Left Chart: X-axis is Subject Count (2 to 10), Y-axis is DINOv2 score. Plot three lines for MOSAIC, XVerse, and PSR. Right Chart: X-axis is Subject Count, Y-axis is SCR@0.4. Plot three lines for the models. The left chart should show a steep downward trend, and the right chart should show a steep upward trend approaching 1.0.)*

#### Catastrophic Identity Forgetting
The most striking finding from our benchmark is the catastrophic collapse of identity fidelity when scaling beyond 4 subjects. As shown in the DINOv2 scores, all models experience a precipitous drop. For instance, MOSAIC, which performs best at the 2-subject level with a DINOv2 score of 0.424, plummets to 0.110 at 10 subjects. XVerse and PSR follow a nearly identical trajectory, dropping from 0.355 and 0.324 to ~0.10, respectively. 

This failure is further magnified when examining the Subject Collapse Rate (SCR). At the most lenient threshold ($\tau=0.4$), MOSAIC exhibits a 48.8% collapse rate even with just 2 subjects. When pushed to 8 subjects, the SCR skyrockets to 94.7% for MOSAIC, 94.4% for XVerse, and 95.0% for PSR. At 10 subjects, the SCR for all models exceeds 96%, indicating that nearly every generated subject has lost its intended identity. This quantitative evidence strongly supports our hypothesis: current attention-routing mechanisms are fundamentally inadequate for resolving dense physical entanglement.

#### Model Comparison
While all models fail at extreme subject counts, **MOSAIC** demonstrates a noticeably stronger baseline in low-complexity scenarios. At 2 subjects, MOSAIC achieves the highest DINOv2 score (0.424) and the lowest SCR (48.8%), outperforming XVerse (0.355 / 58.8%) and PSR (0.324 / 63.3%). This suggests that MOSAIC's representation-centric disentanglement provides a more robust defense against attention leakage when the scene is relatively sparse. However, this advantage quickly dissipates as the subject count reaches 6 and beyond, underscoring the universal scalability bottleneck in current DiT/FLUX architectures.

### 5.2. The Trade-off: Semantic vs. Physical Disentanglement

A counter-intuitive phenomenon emerges when we analyze the Text-Image Alignment (CLIP-T) scores alongside the identity metrics. While DINOv2 scores plummet as the subject count increases, the CLIP-T scores for all models actually exhibit a slight upward trend. For example, PSR's CLIP-T score rises from 0.274 (2 subjects) to 0.309 (10 subjects), and MOSAIC rises from 0.261 to 0.299.

This exposes a critical trade-off mechanism inherent in current diffusion models. When faced with the overwhelming complexity of generating 8 or 10 distinct, interacting individuals, the models actively "take a shortcut." Instead of attempting to faithfully reconstruct each specific identity—which would require precise, uncorrupted attention routing—the models revert to generating a generic "group of people" that satisfies the macro-level semantic constraints of the prompt. Consequently, the global CLIP-T score improves, but at the total expense of personalized identity.

### 5.3. Qualitative Failure Analysis

To better understand *how* these models fail, we perform a qualitative visual analysis of the generated scenes.

*(Placeholder: Figure 3. Insert a grid of generated images comparing 2-subject generation vs. 8-subject generation across the three models. Show successful 2-subject images on the left, and completely failed 8-subject images on the right. Highlight specific failure regions with red bounding boxes.)*

Visual inspection corroborates our quantitative findings and reveals three primary failure modes during multi-subject interaction:
1.  **Identity Bleeding**: Features from one subject (e.g., clothing color, facial structure, glasses) seamlessly bleed into an adjacent subject, especially during physical contact (e.g., hugging).
2.  **Homogenization**: In dense scenes (8-10 subjects), the model often generates multiple copies of the *same* dominant reference identity, completely ignoring the other requested subjects.
3.  **Amodal Collapse**: When subjects are occluded, the model frequently fails to render the hidden geometry correctly, resulting in fused limbs or disembodied floating heads.