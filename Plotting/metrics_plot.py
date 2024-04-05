import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
categories1 = ["BRCA precision", "LUNG precision", "BRCA recall", "LUNG recall", "BRCA F1", "LUNG F1"]
categories2 = ["BRCA precision", "SKCM precision", "BRCA recall", "SKCM recall", "BRCA F1", "SKCM F1"]
categories3 = ["LUNG precision", "SKCM precision", "LUNG recall", "SKCM recall", "LUNG F1", "SKCM F1"]
categories4 = ["BRCA precision", "LUNG precision", "SKCM precision", "BRCA recall", "LUNG recall", "SKCM recall", "BRCA F1", "LUNG F1", "SKCM F1"]

# Generate random data for each category and each model
data1 = np.array([[0.734, 0.940, 0.944, 0.719, 0.824, 0.813],
                  [0.813, 0.969, 0.968, 0.817, 0.883, 0.885],
                  [0.806, 0.978, 0.978, 0.806, 0.883, 0.882],  
                  [0.824, 0.951, 0.948, 0.830, 0.880, 0.884]])
data2 = np.array([[0.719, 0.905, 0.981, 0.348, 0.830, 0.500],
                  [0.907, 0.992, 0.996, 0.824, 0.949, 0.899],
                  [0.907, 0.992, 0.996, 0.824, 0.949, 0.899], 
                  [0.908, 0.983, 0.991, 0.828, 0.948, 0.898]])
data3 = np.array([[0.800, 0.981, 0.996, 0.464, 0.887, 0.620],
                  [0.943, 0.952, 0.979, 0.874, 0.961, 0.910],
                  [0.950, 0.904, 0.950, 0.889, 0.948, 0.890],
                  [0.948, 0.937, 0.971, 0.886, 0.959, 0.910]])
data4 = np.array([[0.568, 0.969, 0.954, 0.973, 0.702, 0.356, 0.717, 0.812, 0.514],
                  [0.760, 0.938, 0.975, 0.933, 0.830, 0.828, 0.837, 0.879, 0.894],
                  [0.743, 0.975, 0.985, 0.976, 0.804, 0.8175, 0.842, 0.879, 0.892], 
                  [0.762, 0.950, 0.975, 0.946, 0.830, 0.828, 0.843, 0.884, 0.895]])
data1_std = np.array([[0.0708, 0.0248, 0.0199, 0.0921, 0.049, 0.065],
                      [0.0707, 0.0212, 0.0210, 0.0788, 0.0469, 0.0518],
                      [0.0727, 0.0142, 0.0129, 0.0839, 0.0469, 0.0542],
                      [0.0846, 0.0203, 0.0192, 0.0911, 0.0536, 0.0582]])
data2_std = np.array([[0.0279, 0.0733, 0.0139, 0.0821, 0.0232, 0.0971],
                      [0.0282, 0.0065, 0.0031, 0.0568, 0.0162, 0.0352],
                      [0.0274, 0.0065, 0.0031, 0.0552, 0.0157, 0.0339],
                      [0.0284, 0.0173, 0.0094, 0.0578, 0.0134, 0.0310]])
data3_std = np.array([[0.0364, 0.0118, 0.00201, 0.121, 0.0224, 0.117],
                      [0.0193, 0.0300, 0.0132, 0.0443, 0.0122, 0.0286],
                      [0.0303, 0.0764, 0.0431, 0.0732, 0.0133, 0.0248],
                      [0.0189, 0.0460, 0.0219, 0.0438, 0.013, 0.0375]])
data4_std = np.array([[0.0369, 0.0272, 0.0437, 0.0201, 0.0837, 0.0825, 0.0338, 0.0620, 0.0934],
                      [0.0553, 0.0203, 0.00974, 0.0226, 0.0707, 0.0602, 0.0369, 0.0452, 0.0326],
                      [0.0611, 0.0148, 0.00959, 0.0134, 0.0854, 0.0533, 0.0423, 0.0554, 0.0316],
                      [0.067, 0.0178, 0.0117, 0.0188, 0.0863, 0.0528, 0.0455, 0.0547, 0.0274]])

# Plotting
models = ['LDA', 'Logistic Regression', 'SVM', 'Random Forest']
width = 0.8
x1 = np.arange(len(categories1))
x2 = np.arange(len(categories2))
x3 = np.arange(len(categories3))
x4 = np.arange(len(categories4))

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for i in range(data1.shape[0]):
    axs[0,0].bar(x1 - width/2 + i * width/data1.shape[0], data1[i, :], width/data1.shape[0], yerr=data1_std[i, :], label=f'Data {i+1}', zorder=3)
    axs[0,0].set_xticks(x1)
    axs[0,0].set_xticklabels(categories1, rotation=15)
    axs[0,0].legend(labels=models, loc='lower left')
    axs[0,0].set_title("BRCA VS LUNG")
    axs[0,0].set_ylabel("Value of metrics")
    axs[0,0].grid(True, axis='y', zorder=0)
    

for i in range(data2.shape[0]):
    axs[0,1].bar(x2 - width/2 + i * width/data2.shape[0], data2[i, :], width/data2.shape[0], yerr=data2_std[i, :], label=f'Data {i+1}', zorder=3)
    axs[0,1].set_xticks(x2)
    axs[0,1].set_xticklabels(categories2, rotation=15)
    axs[0,1].legend(labels=models, loc='lower left')
    axs[0,1].set_title("BRCA VS SKCM")
    axs[0,1].set_ylabel("Value of metrics")
    axs[0,1].grid(True, axis='y', zorder=0)
    
    
for i in range(data3.shape[0]):
    axs[1,0].bar(x3 - width/2 + i * width/data3.shape[0], data3[i, :], width/data3.shape[0], yerr=data3_std[i, :], label=f'Data {i+1}', zorder=3)
    axs[1,0].set_xticks(x3)
    axs[1,0].set_xticklabels(categories3, rotation=15)
    axs[1,0].legend(labels=models, loc='lower left')
    axs[1,0].set_title("LUNG VS SKCM")
    axs[1,0].set_ylabel("Value of metrics")
    axs[1,0].grid(True, axis='y', zorder=0)
    
for i in range(data4.shape[0]):
    axs[1,1].bar(x4 - width/2 + i * width/data4.shape[0], data4[i, :], width/data4.shape[0], yerr=data4_std[i, :], label=f'Data {i+1}', zorder=3)
    axs[1,1].set_xticks(x4)
    axs[1,1].set_xticklabels(categories4, rotation=20)
    axs[1,1].legend(labels=models, loc='lower left')
    axs[1,1].set_title("BRCA VS LUNG VS SKCM")
    axs[1,1].set_ylabel("Value of metrics")
    axs[1,1].grid(True, axis='y', zorder=0)

fig.suptitle("Comparison of metrics for different models and datasets")

plt.tight_layout()
plt.show()
