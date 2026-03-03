# HDECGCN-Framework
X. Liu, S. Liu, W. Chen and S. Qu, "HDECGCN: A Heterogeneous Dual Enhanced Network Based on Hybrid CNNs Joint Multiscale Dynamic GCNs for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-17, 2024, Art no. 5515717, doi: 10.1109/TGRS.2024.3387420.
The execution steps of the entire framework code are as follows:
(1) Run the DynamicParametersAndDualSuperpixel.m code to get the IP/UP/KSC_gyh.mat, final.mat, tepos.mat, and trpos.mat files, for example, we have saved the data samples in the directory 'HDECGCN-Framework/data/..' after running the DynamicParametersAndDualSuperpixel.m.
(2) Run Decisionfusion. py to obtain the final three classification results OA, AA, and Kappa.
Don't forget to modify the path of the data file in the code.
If you have any questions about the code, do not hesitate to contact xuanliu2612@163.com.
