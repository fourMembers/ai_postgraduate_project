import os
import numpy as np
import matplotlib.pyplot as plt



train_path = os.path.join(os.path.join(os.getcwd(), "data",  "Task07_Pancreas", "imagesTr"))
#train_path = os.path.join(os.path.join(os.getcwd(), "data",  "resized_dataset", "imagesTr"))
all_data = os.listdir(train_path)
rm=[]

for i in range(len(all_data)):
  if all_data[i][0]==".":
    rm.append(all_data[i])

for element in rm:
    all_data.remove(element)



from nibabel.testing import data_path
from nilearn import plotting
import nibabel as nib

x_size = []
y_size = []
z_size = []
for data in all_data:
    img = nib.load(os.path.join(train_path, data))
    x_size.append(int(img.shape[0]))
    y_size.append(int(img.shape[1]))
    z_size.append(int(img.shape[2]))



plt.bar(range(len(x_size)),x_size)
plt.show()


plt.bar(range(len(y_size)),y_size)
plt.show()



plt.bar(range(len(z_size)),z_size)
plt.show()


img = nib.load(os.path.join(train_path, all_data[-11]))
array = np.array(img.dataobj)
img.shape
plotting.plot_anat(img)
np.mean(z_size)
np.median(z_size)
array.shape
resized_array = np.resize(array,(512,512,93))
resized_array.shape
new_image = nib.Nifti1Image(resized_array,affine=np.eye(4))





