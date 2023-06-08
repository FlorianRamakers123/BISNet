# Boundary Interactive Segmentation Network (BISNet)

## Abstract
The use of deep learning for medical image segmentation has proven to be beneficial in many occasions.
However, automatic segmentation techniques are less suitable for clinical settings due to insufficient clinical acceptability and potential medical liabilities.
Interactive segmentation solves these problems by allowing the clinicians to refine the automatic segmentation.
In this thesis *Boundary Interactive Segmentation Network (BISNet)* is proposed for refinement of automatic 3D segmentations using boundary clicks.
A boundary click provides an indication of where the user prefers the boundary of the segmented volume to be.
In a preliminary study, BISNet is completely tuned for optimal accuracy and responsiveness.
To achieve the desired level of performance, new loss functions, guidance dropout techniques and responsiveness metrics are proposed.
In a subsequent clinical study, it is compared to current state-of-the-art interactive segmentation framework *DeepEdit* and to the current clinical protocol for manual segmentation.
Based on the experiments with three clinicians with varying segmentation experience, it is shown that BISNet allows to create biologically plausible segmentations with 100\% clinical acceptability in significantly less time (reduction of ± 81s) and number of interactions than manual segmentation and DeepEdit and with lower perceived workload.
Furthermore, refined segmentations achieved dice scores similar to the intra-observer variability of a clinical segmentation expert.
This implies that BISNet allows clinicians with zero experience in segmentation to achieve the same level of quality as a segmentation expert.
Consequently it is concluded that BISNet improves both upon the state-of-the-art and manual segmentation in terms of speed, accuracy and perceived workload and is independent of user experience.


## BISNet in action
<video width="1280" height="600" controls>
  <source src="BISNet_video.mp4" type="video/mp4">
</video>


## BISNet in 3D Slicer
WIP