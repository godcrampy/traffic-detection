from PIL import Image, ImageDraw
import numpy as np
import cv2


img_pil = Image.open('test.jpg')
img_np = np.array(img_pil)
img_pil = Image.fromarray(img_np)


draw = ImageDraw.Draw(img_pil)
draw.line([0, 0, 1024, 636], fill = (255,0,0), width = 4)
del draw
img_np = np.array(img_pil)
img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

cv2.imshow('image',cv2.resize(img_np, (1024,636)))
cv2.waitKey(0)
cv2.destroyAllWindows()