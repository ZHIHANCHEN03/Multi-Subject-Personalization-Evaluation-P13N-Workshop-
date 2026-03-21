## 6. Discussion

Our comprehensive evaluation uncovers several fundamental bottlenecks in current multi-subject personalization models, pointing to critical areas for future research within the generative AI community.

### 6.1. The "Semantic Shortcut" Hypothesis
The divergent trends between CLIP-T [3] (which slightly improves) and DINOv2 [4] (which drastically drops) as subject count increases suggest that models are not merely "failing" to generate images, but are actively optimizing for the wrong objective. In DiT [11] and FLUX [12] architectures, when the global attention mechanism is overwhelmed by multiple, competing identity embeddings, the model defaults to the strongest available prior: the global text prompt. We term this the "semantic shortcut." The model synthesizes a structurally plausible scene of "many people" to satisfy the text encoder, completely washing out the localized, high-frequency details required for identity preservation. This highlights a dangerous flaw in relying solely on CLIP for generative evaluation.

### 6.2. Physical Disentanglement vs. Semantic Disentanglement
While recent methods like MOSAIC [5] and XVerse [6] have made significant strides in semantic disentanglement (e.g., ensuring "dog" features don't mix with "cat" features in latent space), our benchmark reveals that they still fail at *physical disentanglement*. When subjects interact (e.g., hugging) or occlude one another, the boundary between their spatial regions becomes blurred. Because current models lack explicit 3D reasoning or amodal completion mechanisms [14], the physical overlap directly translates into attention leakage in the 2D feature maps. Solving "who is who" is insufficient if the model cannot resolve "who is in front of whom."

### 6.3. Limitations
We acknowledge certain limitations in our current benchmark. First, our evaluation is primarily focused on human and animal subjects; evaluating multi-object compositions involving inanimate objects with rigid geometries may yield different failure modes. Second, our scene types (neutral, occlusion, interaction) are defined via prompt engineering rather than explicit 3D layout controls [9, 10], meaning the severity of occlusion is left to the model's interpretation.

---

## 7. Conclusion

In this paper, we presented a rigorous stress-test benchmark to evaluate the limits of subject-driven diffusion models in multi-entity compositions, directly addressing the P13N workshop's call for robust evaluation metrics. We demonstrated that standard CLIP-based metrics are dangerously blind to local identity collapse, and we proposed the Subject Collapse Rate (SCR) based on DINOv2 to quantify this failure. Our extensive evaluation of SOTA models (MOSAIC [5], XVerse [6], PSR [7]) yielded a sobering conclusion: while models can handle 2-4 isolated subjects, they suffer from catastrophic identity forgetting—with SCR approaching 100%—when forced to compose 6 to 10 interacting identities. This failure exposes a fundamental flaw in current global attention routing mechanisms, which sacrifice localized physical fidelity for macro-level semantic alignment. We hope our benchmark and the proposed SCR metric will serve as a clear, demanding target for the next generation of physically-grounded personalization models.
