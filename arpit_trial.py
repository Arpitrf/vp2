import h5py
import numpy as np
import matplotlib.pyplot as plt

# f = h5py.File('/home/arpit/test_projects/vp2/vp2/robosuite_benchmark_tasks/combined/rendered_256.hdf5', "r") 
# demos = f["data"].keys()
# print(f['data'].keys(), len(f['data'].keys()))
# print(f['data']['demo_1'].keys())
# print(f['data']['demo_1']['obs'].keys())
# print(np.array(f['data']['demo_1']['obs']['agentview_shift_2_seg']).shape)
# imgs = np.array(f['data']['demo_1']['obs']['agentview_shift_2_seg'])
# fig, ax = plt.subplots(2,2)
# ax[0][0].imshow(imgs[0])
# ax[0][1].imshow(imgs[5])
# ax[1][0].imshow(imgs[10])
# ax[1][1].imshow(imgs[15])
# plt.show()

# -------------------------------------------

f = h5py.File('/home/arpit/test_projects/vp2/vp2/robosuite_benchmark_tasks/5k_slice_rendered_256.hdf5', "r")
print("f.keys(): ", f['mask'].keys()) 
print(f['mask']['train'])
print(type(np.array(f['mask']['valid'])[0]))
# print("mask: ", np.array(f['mask']['valid']))
# demos = f["data"].keys()
# print(len(f['data'].keys()))
# print(f['data']['demo_1'].keys())
# print(f['data']['demo_1']['obs'].keys())
# print(np.array(f['data']['demo_1']['actions']).shape)
# print(f['data']['demo_1']['dones'])
# print(f['data']['demo_1']['rewards'])
# print(f['data']['demo_1']['states'])
# print(np.array(f['data'][f'demo_1']['obs']['agentview_shift_2_seg']).shape)
# imgs = np.array(f['data']['demo_1']['obs']['agentview_shift_2_seg'])
# fig, ax = plt.subplots(2,2)
# ax[0][0].imshow(imgs[0])
# ax[0][1].imshow(imgs[5])
# ax[1][0].imshow(imgs[10])
# ax[1][1].imshow(imgs[15])
# plt.show()