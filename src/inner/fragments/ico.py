import cv2
import numpy as np
import matplotlib.pyplot as plt

def ICO(img):
    img = img.astype(np.float32)

    # Gradient
    gx = cv2.Sobel(img, cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F,0,1,ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.rad2deg(np.arctan2(gy,gx)) % 180
    mag /= mag.max()

    # Non-Maximum Suppression
    nms = np.zeros_like(mag)
    r,c = mag.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            q=r_=0
            if (0<=ang[i,j]<22.5) or (157.5<=ang[i,j]<=180): q,r_=mag[i,j+1],mag[i,j-1]
            elif (22.5<=ang[i,j]<67.5): q,r_=mag[i+1,j-1],mag[i-1,j+1]
            elif (67.5<=ang[i,j]<112.5): q,r_=mag[i+1,j],mag[i-1,j]
            else: q,r_=mag[i-1,j-1],mag[i+1,j+1]
            if mag[i,j]>=q and mag[i,j]>=r_: nms[i,j]=mag[i,j]

    # Adaptive thresholds
    high = np.mean(nms)+np.std(nms)
    low  = 0.5*high
    edges = np.where(nms>=high,1,np.where(nms>=low,0.5,0))

    # Hysteresis
    for i in range(1,r-1):
        for j in range(1,c-1):
            if edges[i,j]==0.5 and np.any(edges[i-1:i+2,j-1:j+2]==1):
                edges[i,j]=1
            elif edges[i,j]==0.5:
                edges[i,j]=0
    return edges

# Example
img = cv2.imread("mammogram_filtered.png",0)
edges = ICO(img)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.title("Input"); plt.imshow(img,cmap="gray"); plt.axis("off")
plt.subplot(1,2,2); plt.title("ICO Edges"); plt.imshow(edges,cmap="gray"); plt.axis("off")
plt.show()