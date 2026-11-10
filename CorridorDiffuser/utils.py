from matplotlib import pyplot as plt


def plot_1_1(data):
    data = (data.detach().cpu().numpy() + 1) / 2 * 255
    plt.imshow(data)
    plt.colorbar()
    plt.show()


def plot_0_1(data):
    data = data.detach().cpu().numpy()  * 255
    plt.imshow(data)
    plt.colorbar()
    plt.show()


from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = Image.open('data/maps/map_0000.png').convert('RGB')
img_np = np.array(img, dtype=np.float32)
plt.imshow(img_np, origin='lower')
plt.show()


img = Image.open('data/targets/target_0000_0.png').convert('L')
img_np = np.array(img, dtype=np.float32)/255
plt.imshow(img_np, origin='lower')
plt.colorbar()
plt.show()
