import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Let's assume 'A' is your image array with shape (45, 2, 256, 256)
# You need to adjust your code to visualize one image at a time

A = np.load("test_masks.npy")
single_image = A[2, 1]  # This slices out one image of shape (256, 256)

plt.imshow(single_image, cmap='gray')  # 'cmap' can be adjusted based on your data
plt.show()

# Assuming 'A' is your image array with the shape (45, 2, 256, 256)

# # Calculate the total number of images
# num_sets, num_images_per_set, _, _ = A.shape

# # Create a figure with subplots
# fig, axs = plt.subplots(nrows=num_sets, ncols=num_images_per_set, figsize=(num_images_per_set * 4, num_sets * 4))

# for set_idx in range(num_sets):
#     for img_idx in range(num_images_per_set):
#         ax = axs[set_idx, img_idx] if num_sets > 1 else axs[img_idx]
#         ax.imshow(A[set_idx, img_idx], cmap='gray')  # Assuming grayscale images
#         ax.axis('off')  # Hide axes
#         ax.set_title(f"Set {set_idx + 1}, Image {img_idx + 1}")

# plt.tight_layout()
# plt.show()
