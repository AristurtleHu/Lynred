# Lynred - Restoring Infrared Imagery

A high-performance pipeline designed to detect and restore heavily degraded vertical column defects.

It is engineered during my R&D internship with **LYNRED** for raw 16-bit infrared image sequences across VGA, SXGA, and HD sensors. 

Our solution achieved the **top one** comprehensive performance among all participating teams, outperforming peers in both strict **quantitative benchmarks** and **qualitative visual fidelity**.

## Defective Columns Restoration

In infrared imaging systems, read-out integrated circuit (ROIC) anomalies (such as operational amplifier sensitivity drift) frequently introduce severe **vertical stripe defects**. These manifest as three primary distinct typologies:
*   **Noisy columns:** Constant high temporal noise compared to the background.
*   **Blinking columns:** Sudden high/low-frequency digital level shifts over time.
*   **Fragmented blinking columns:** Rapid spatiotemporal structural breaks where defects randomly appear or vanish within a single column frame-by-frame.

Our objective was to optimize a dual-metric final score balancing localization accuracy ($$\text{F1-Score}$$) and restoration fidelity ($$\text{RMSE}$$).

### 🛠️ System Architecture & Methodology

Instead of relying purely on deep learning or simple traditional interpolation, we designed a hybrid framework combining **Advanced Statistical Anomaly Detection** and **Localized Edge-Preserving Variational Filtering**.

#### 1. Defect Detection Engine
We developed and benchmarked five independent localization approaches:
*   **Baseline Statistical Detectors:** Difference-in-Median filtering and Temporal Standard Deviation mapping ($$\text{F1} \approx 0.50$$).
*   **Unsupervised Anomaly Detection:** Isolation Forests using engineered statistical descriptors (Temporal Mean, Spatial Variance, Differential Context).
*   **Deep Learning (2D Asymmetric CNN):** To tackle highly non-linear fragmented blinking, we built a 2D CNN with an elongated vertical receptive path ($$7 \times 3$$ and $$5 \times 3$$ asymmetric kernels) to prevent horizontal cross-talk. Coupled with multi-scale morphological area constraints, it achieved a peak aggregate **$$\text{F1}$$-score of 0.89**.
*   **Final Production Layer (Local Mean of Medians):** A robust uniform-filtered column median comparison algorithm. It achieved the highest comprehensive system stability across multi-resolution data sensors.

<div align="center">
  <img src="docs/image/detection_local_mean.png" alt="Column Defect Detection Threshold" />
  <p>🔍 <em>Figure 1: Adaptive statistical threshold alignment via Detector 5 (Local Mean of Medians).</em></p>
</div>

#### 2. Signal Decomposition & Correction Pipeline
We formulated the restoration task strictly as a signal decomposition problem rather than blind pixel replacement:
$$X_{\text{corrected}} = X_{\text{raw}} - S$$

Where $$S$$ isolates the additive vertical stripe noise component. This math guarantee ensures that healthy pixels remain absolutely untouched.

The core correction execution follows a 4-stage 1D pipeline applied independently to each corrupt column:
1.  **Guided Reference Estimation:** Distance-weighted spatial interpolation from the nearest $$n=3$$ clean columns.
2.  **1D Side-Window Filtering (SWF):** Sub-window boundary checking ($$r=8$$) to prevent blurring physical scene edges.
3.  **High-Frequency Isolation:** Separating fine texture and stripe patterns.
4.  **1D Gradient Domain Guided Image Filter (GDGIF):** Minimizing a local linear optimization model with edge-perception regularization ($$\mu = 0.022$$).

<div align="center">
  <img src="docs/image/output_local_mean.png" alt="Before and After Correction" />
  <p>✨ <em>Figure 2: Visual verification of the final pipeline output. Sharp horizontal textures and faint background objects are perfectly preserved with zero smearing.</em></p>
</div>

### 📊 Performance & Experimental Results

The combined **Local Mean of Medians + Side-Window GDGIF** architecture achieved an outstanding **average final score of 0.75** on the multi-sensor challenge dataset. 

#### Comprehensive Visual Residual Evaluation
To thoroughly assess artifact injection, ringing, and edge-degradation, we mapped the absolute pixel-wise error residuals ($$|X_{\text{corrected}} - GroundTruth|$$) across all candidate approaches.

<div align="center">
  <img src="docs/image/vertical_analysis.png" alt="Visual Comparison of Correction Methods"/>
  <p>📊 <em>Figure 3: Comprehensive residual diagnostics map across all tested recovery methods. Lower and cleaner residual bounds indicate superior correction fidelity without introducing structural ghosting.</em></p>
</div>

#### Quantitative Comparison of Correction Models

| Correction Method        | Evaluation Score | Peak Absolute Residual | Key Operational Limitation                               |
| :----------------------- | :--------------: | :--------------------: | :------------------------------------------------------- |
| **Side-Window 1D GDGIF** |    **0.7035**    |      **~40 LSB**       | *(Selected) Optimal edge preservation*                   |
| Guided 1D Filter         |      0.6733      |        ~80 LSB         | Slightly weaker edge preservation than SWF               |
| Fourier Notch Filtering  |      0.4779      |        ~60 LSB         | Gibbs ringing artifacts; fails on non-periodic blinking  |
| 2D Spatial Guided Filter |      0.4384      |        ~60 LSB         | Vertical incoherence from pixel-wise guide construction  |
| Neighbor Interpolation   |      0.4601      |        ~100 LSB        | Blind low-pass behavior smears diagonal scene structures |
| Temporal Mean Averaging  |      0.1974      |        ~250 LSB        | Severe ghosting artifacts under dynamic camera motion    |

### 💡 Key Takeaways & Engineering Insights

1.  **1D vs. 2D Constraints:** Attempting 2D spatial patch corrections (`guided_2d`, `side_window_2d`) introduced severe vertical incoherence because horizontal image structures leaked into the vertical linear models. Reverting to an explicit 1D column-wise tracking solved this fundamentally.
2.  **Edge Preservation:** Traditional spatial filters acting as blind low-pass smoothers always smear sharp boundaries. By implementing the Side-Window constraint, the filter gracefully handles high-contrast intersections without introducing ringing or artifacts.
3.  **Edge Deployment Ready:** The entire inference pipeline operates on local 1D data vectors, entirely bypassing heavy global matrix operations—making it exceptionally suited for embedded edge infrared sensors or real-time low-latency systems.

*Details in the [report](https://github.com/AristurtleHu/Lynred/blob/main/docs/lynred.pdf).*
