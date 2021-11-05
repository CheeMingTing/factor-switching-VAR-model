# f-SVAR

Estimating dynamic connectivity states in neuroimaging data using regime-switching factor models

f-SVAR (Factor switching vector autoregressive (VAR) model) Toolbox is a Matlab package to estimate dynamic, directed functional connectivity states in high-dimensional neuroimaging time series data (fMRI, EEG etc). The underlying method uses a regime-switching factor model. The script example_simulation.m generates random data from a 2-state switching VAR(1) model, estimates the f-SVAR model, and plots estimates of latent state sequence and coefficient matrices as compared to the ground-truth. The script example_fmri.m loads resting-state fMRI time series data of three subjects, produces f-SVAR estimators, and plot the estimated state sequence for each subject, and common state connectivity matrices of 96 regions of interest (ROIs) partitioned into six resting-state networks.

For more detailed description and applications, please refer to the User Manual and references:

1.	C.-M. Ting, H. Ombao, S. B. Samdin and Sh-H. Salleh, “[Estimating dynamic connectivity states in fMRI using regime-switching factor models](https://ieeexplore.ieee.org/document/8166781),” IEEE Trans. Med. Imag., 37(4), pp. 1011 - 1023, 2018.
2.	S. B. Samdin, C.-M. Ting, H. Ombao, and Sh-H. Salleh, “A unified estimation framework for state-related changes in effective brain connectivity,” IEEE Trans. Biomed. Eng., vol. 64, no. 4, pp. 844–858, 2017.
3.	C.-M. Ting, A.-K. Seghouane, S.-H. Salleh, and A. M. Noor, “Estimating effective connectivity from fMRI data using factor-based subspace autoregressive models,” IEEE Sig. Process. Lett., vol. 22, no. 6, pp. 757–761, 2014.
