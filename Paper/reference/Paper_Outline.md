# Multi-Subject Personalization Evaluation Paper Outline (CVPR 2026 Format)

*Target: P13N Workshop @ CVPR 2026*
*Format: Double-blind, Max 8 pages (Long Paper) or 4 pages (Short Paper), excluding references.*

---

## 0. Abstract (摘要)
*   **字数限制**: 严格限制在一小段（通常不超过 200 词）。
*   **写作重点**:
    1.  **背景**: 个性化生成模型（Subject-driven Diffusion Models）目前取得了很大进展，但在多主体（Multi-Subject）场景下表现如何尚不清楚。
    2.  **核心痛点**: 缺乏系统性的 Benchmark 和能够敏锐捕捉“身份崩溃”的评估指标。
    3.  **我们做了什么 (Contribution)**: 我们构建了一个包含 2-10 个主体的极限压力测试 Benchmark，并引入了 DINOv2 相似度和 SCR (Subject Collapse Rate) 指标。
    4.  **核心结论**: 评估了 3 个 SOTA 模型（MOSAIC, XVerse, PSR），发现随着主体增加，模型发生了灾难性的身份遗忘（SCR 高达 90% 以上），揭示了文本对齐与身份保真度之间的 Trade-off。

## 1. Introduction (引言)
*   **篇幅建议**: 约 1 页。
*   **排版建议**: 在第一页右上角放置 **Figure 1 (Teaser)**。Teaser 图应包含一组强烈的对比：比如 MOSAIC 在 2 个主体时生成的精美图像 vs. 在 8 个主体时生成的糊脸/身份混淆的图像。
*   **写作重点**:
    1.  **大背景**: 介绍 Diffusion 模型在 Personalization 上的成功（如 DreamBooth, LoRA 等）。
    2.  **指出 Gap**: 现有研究大多集中在单主体或双主体，真实世界需要多实体互动。
    3.  **现有评估的缺陷**: 传统 CLIP 指标对局部身份特征不敏感，无法反映多主体生成的真实失败率。
    4.  **本文贡献 (Contributions list)**:
        *   提出了一个分梯度的多主体 Benchmark (2-10 subjects)。
        *   引入了更严格的评估体系：DINOv2 + SCR。
        *   深入分析了 3 个 SOTA 模型，揭示了它们在多主体组合时的灾难性失败现象。

## 2. Related Work (相关工作)
*   **篇幅建议**: 约 0.5 - 0.75 页。
*   **写作重点**:
    1.  **Subject-driven Image Generation**: 简述 DreamBooth, Textual Inversion, Custom Diffusion 等单主体方法。
    2.  **Multi-subject Composition**: 重点介绍你评测的基线模型（MOSAIC, XVerse, PSR）以及其他相关工作（如 CustomNet, Cones 等），讨论它们在处理多主体时采用的方法（如 Attention Control, 区域约束等）。
    3.  **Evaluation Metrics for Personalization**: 讨论现有的评估指标（如基于 CLIP 的相似度），并指出它们的局限性。引出 DINOv2 在特征匹配上的优势。

## 3. Multi-Subject Benchmark (多主体基准测试构建)
*   **篇幅建议**: 约 0.5 - 1 页。
*   **写作重点**:
    1.  **Subject Pool**: 介绍你如何从现有的 XVerse 和 COSMISC 数据集中筛选和整合 subject reference images。
    2.  **Prompt Design**: 详细说明你设计的 5 个难度级别（2, 4, 6, 8, 10 subjects）。
    3.  **Scene Types**: 介绍你的 prompt 分类（Interaction, Occlusion, Neutral），并给出具体的 Prompt 例子（可以使用表格展示）。
    4.  **Data Generation Setup**: 简述每个 prompt 使用 3 个随机种子，共生成了多少张测试图。

## 4. Evaluation Metrics (评估指标体系)
*   **篇幅建议**: 约 0.75 页。
*   **写作重点** (这是本文的核心亮点之一，必须写清楚公式和动机):
    1.  **Text-Image Alignment (CLIP-T)**: 简述如何使用 CLIP Text Encoder 衡量整体语义一致性。
    2.  **Identity Preservation (DINOv2)**: **重点论述** 为什么抛弃传统的 CLIP Image-to-Image，转而使用 DINOv2（强调其对局部特征和结构的敏感性）。
    3.  **Subject Collapse Rate (SCR)**: **重点论述** 这个新指标。给出公式：$SCR = \frac{\text{collapsed\_subjects}}{\text{total\_subjects}}$。明确定义什么是“崩溃”（例如 DINOv2 余弦相似度 $< \tau$），并说明你选取的阈值（$\tau \in \{0.4, 0.5, 0.6\}$）。

## 5. Experiments (实验与结果分析)
*   **篇幅建议**: 约 2 - 3 页。包含大量图表。
*   **排版建议**: 放置 **Table 1** (综合数值表) 和 **Figure 2** (折线趋势图：Subject Count vs. DINOv2 / SCR)。
*   **写作重点**:
    1.  **Implementation Details**: 交代运行环境、DINOv2 和 CLIP 的具体模型版本等。
    2.  **Quantitative Results (定量分析)**:
        *   **Catastrophic Forgetting in Multi-subject**: 引用数据，指出人数从 2 增加到 10 时，DINOv2 分数断崖式下跌，SCR 飙升至近 100%。
        *   **Model Comparison**: 对比 MOSAIC, XVerse, PSR 的表现，指出 MOSAIC 在较少主体（2-4）时表现最佳，但在极端场景下大家都失败了。
    3.  **Qualitative Results (定性分析/视觉展示)**:
        *   展示一系列失败案例（Failure Cases）。
        *   分析失败的具体表现：比如“身份特征混淆 (Identity Bleeding)”、“某个主体完全消失 (Missing Subject)”、“全员长得一样 (Homogenization)”。

## 6. Discussion (讨论)
*   **篇幅建议**: 约 0.5 页。
*   **写作重点**:
    1.  **The Trade-off between Text Alignment and Identity**: 讨论为什么随着人数增加，身份完全崩溃了，但 CLIP-T（文本分数）反而微微上升。解释模型“走捷径”的现象。
    2.  **Impact of Scene Types**: (可选) 如果你的数据支持，简要讨论 interaction 场景是否比 neutral 场景更容易导致主体崩溃。
    3.  **Limitations & Future Work**: 指出目前评估的一点局限性（例如仅测试了 3 个模型，或者主要集中在人/动物），并呼吁社区开发更好的多主体注意力隔离机制。

## 7. Conclusion (结论)
*   **篇幅建议**: 一小段。
*   **写作重点**: 总结全文。再次重申：现有的 Subject-driven 模型在多实体组合上存在严重缺陷；本文提出的 Benchmark 和 SCR 指标为未来的研究提供了一个明确的靶子（Target）。

## References (参考文献)
*   不计入页数限制。使用 CVPR 模板自带的 `ieeenat_fullname.bst` 格式。

---
**提交前 Checklist (CVPR Double-blind 规范):**
- [ ] 确保 `.tex` 导言区使用 `\usepackage[review]{cvpr}`。
- [ ] 检查正文、图表、甚至图片的文件名中是否包含作者姓名或机构名。
- [ ] 检查 Acknowledgements 章节是否已删除或隐藏（致谢会暴露身份）。
- [ ] 检查所有对自身之前工作的引用是否已改为第三人称（如 "Smith et al. [1] proposed..."）。