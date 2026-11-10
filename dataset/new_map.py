from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

src_path = Path(f'data/maps')
dst_path = Path(fr'data/new_maps')

for file in src_path.rglob('*'):
    cor_img = Image.open(file)
    cor_arr = np.array(cor_img)*255
    cv2.imencode('.png', cor_arr)[1].tofile(str(dst_path / file.name))



