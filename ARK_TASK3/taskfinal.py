import cv2
import numpy as np

vid = cv2.VideoCapture("3.mp4")

fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_1 = cv2.VideoWriter("output_3.mp4", fourcc, fps, (700,400))

fgbg = cv2.createBackgroundSubtractorMOG2()

frame_count = 0

prev_rho = None
prev_theta = None
alpha = 0.9
thetas = np.deg2rad(np.arange(180))
cos_t = np.cos(thetas)
sin_t = np.sin(thetas)

def draw_line(img, rho, theta, color):

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a*rho
    y0 = b*rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),color,2)

while True:

    success, image = vid.read()
    if not success:
        break

    frame_count += 1

    fgmask = fgbg.apply(image)

    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(fgmask,50,150)

    if frame_count > 30:

        h,w = edges.shape

        rho_max = int(np.sqrt(h**2 + w**2))

        accumulator = np.zeros((2*rho_max,180),dtype=np.uint64)

        ys,xs = np.where(edges>0)

        for i in range(len(xs)):

            x = xs[i]
            y = ys[i]

            rhos = x*cos_t + y*sin_t
            rhos = np.round(rhos).astype(int) + rho_max

            valid = (rhos>=0) & (rhos<2*rho_max)

            np.add.at(accumulator,(rhos[valid],np.arange(180)[valid]),1)

        flat = accumulator.flatten()
        idx = np.argpartition(flat,-2)[-2:]

        peaks=[]

        for i in idx:

            r = i//180
            t = i%180

            rho = r-rho_max
            theta = thetas[t]

            peaks.append((rho,theta))

        if len(peaks)>=2:

            rho1,theta1 = peaks[0]
            rho2,theta2 = peaks[1]

            rho_mid = (rho1+rho2)/2
            theta_mid = (theta1+theta2)/2

            if abs(theta1-theta2)<np.deg2rad(5):


                prev_rho = rho_mid
                prev_theta = theta_mid

                draw_line(image,rho_mid,theta_mid,(0,0,255))

    image = cv2.resize(image,(700,400))

    out_1.write(image)

vid.release()
out_1.release()
cv2.destroyAllWindows()