from skimage import io, color, filters, morphology, feature, util
from scipy import ndimage
import matplotlib.pyplot as plt

# 1. Loading the image
image = io.imread('../images/inputs/frame_0015.jpg')
gray = color.rgb2gray(image)

# Apply Gaussian blur to reduce noise
blur = filters.gaussian(gray, sigma=1)  # Adjust sigma as needed for smoothing

# Convert to binary image (0-1 range)
bin = util.img_as_ubyte(blur)  # Convert to 8-bit unsigned integer format

# 2. Detect edges first (using Canny)
edges = feature.canny(bin, sigma=2)

# 3. Dilate the edges to connect segments
dilated = morphology.binary_dilation(edges, morphology.disk(2))  # disk(2) dilates by 2 pixels in all directions

# 4. Fill in the edges to create a mask of the crack
cracks_filled = ndimage.binary_fill_holes(dilated)

# 5. Perform skeletonization to find centerline of the filled crack
skeleton = morphology.skeletonize(cracks_filled)

# 6. Display results
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].imshow(gray, cmap='gray')
axes[0].set_title('original')

axes[1].imshow(edges, cmap='gray')
axes[1].set_title('canny edges')

axes[2].imshow(dilated, cmap='gray')
axes[2].set_title('dilated edges')

axes[3].imshow(skeleton, cmap='hot')
axes[3].set_title('skeleton (centerline)')
for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
