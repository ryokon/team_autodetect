# -*- coding: utf-8 -*-

import cv2, os
import numpy
import numpy as np
import matplotlib.pyplot as plt
import imutils
from PIL import Image
from PIL import ImageGrab
import argparse
import glob
from scipy.spatial import distance as dist
import csv
from collections import defaultdict

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}
results = {}
reverse = False


######################


#capture image and configure it to opencv formats, color and gray
cap_img = ImageGrab.grab()
image = np.array(cap_img, 'uint8')
img_rgb = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)

#modify to hsv to define green areas
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
lower_green = np.array([40,50,40])
upper_green = np.array([50,255,255])

#create mask for green area or anything else
mask_inverse1 = cv2.inRange(hsv, lower_green, upper_green)


mask_inverse = mask_inverse1

mask = cv2.bitwise_not(mask_inverse)



#img_rgb applied with mask
res = cv2.bitwise_and(img_rgb,img_rgb, mask = mask_inverse)

res2 = cv2.bitwise_and(img_rgb,img_rgb, mask = mask)

#getting contour
ret,thresh = cv2.threshold(mask_inverse,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]
#contour POINTS
x,y,w,h = cv2.boundingRect(cnt)
#cropping the POINTS out of original picture (img_rgb)
crop_img = img_rgb[y:y+h, x:x+w]
#just resized it
newImg = cv2.resize(crop_img, (210, 315), interpolation=cv2.INTER_LINEAR)



####################


hsv = cv2.cvtColor(newImg, cv2.COLOR_BGR2HSV)
lower_green = np.array([40,60,180])
upper_green = np.array([50,255,255])

lower_yellow = np.array([17,200,40])
upper_yellow = np.array([28,255,255])

lower_blue = np.array([95,165,100])
upper_blue = np.array([115,255,255])



#create mask for green area and the inverse mask for it
mask_inverse1 = cv2.inRange(hsv, lower_green, upper_green)
mask_inverse2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_inverse3 = cv2.inRange(hsv, lower_blue, upper_blue)


mask_inverse = mask_inverse1 + mask_inverse2 + mask_inverse3

mask = cv2.bitwise_not(mask_inverse)

coins_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# make copy of image
coins_and_contours = np.copy(newImg)

# find contours of large enough area
min_coin_area = 1000
large_contours = [cnt for cnt in coins_contours if cv2.contourArea(cnt) > min_coin_area]

bounding_img = np.copy(newImg)

smalls = []
icons = []

# for each contour find bounding box and draw rectangle
for contour in large_contours:
    x, y, w, h = cv2.boundingRect(contour)
    #cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 0, 0), 1)

    smalls.append(cv2.boundingRect(contour))

#cv2.imshow("Show",bounding_img)

######################


for plot in smalls:
	#print(plot)
	x, y, w, h = plot
	icon = bounding_img[y:y+h, x:x+w]

	icons.append(icon)


#######################
#initialize the test icons

histxs = []
ress = []
resizeds = []

grays = []

for icon in icons:
	img_rgbx = icon
	hsvx = cv2.cvtColor(img_rgbx, cv2.COLOR_BGR2HSV)

	lower_green = np.array([40,60,180])
	upper_green = np.array([50,255,255])

	lower_yellow = np.array([17,200,40])
	upper_yellow = np.array([28,255,255])

	lower_blue = np.array([95,165,100])
	upper_blue = np.array([115,255,255])




	mask_inversex = cv2.inRange(hsvx, lower_green, upper_green)
	mask_inverse2 = cv2.inRange(hsvx, lower_yellow, upper_yellow)
	mask_inverse3 = cv2.inRange(hsvx, lower_blue, upper_blue)


	mask_inversef = mask_inversex + mask_inverse2 + mask_inverse3

	maskx = cv2.bitwise_not(mask_inversef)

	white = np.full(img_rgbx.shape, 255, dtype=img_rgbx.dtype)
	background = cv2.bitwise_and(white, white, mask=mask_inversef)
	res1 = cv2.bitwise_and(img_rgbx,img_rgbx, mask = maskx)
	res = cv2.add(res1, background)

	#cv2.imshow("Show",res)

	w, h, c = img_rgbx.shape
	image2 = res[0:h-12, 0:w-6]
	image = image2

	histx = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	histx = cv2.normalize(histx).flatten()


	resized = cv2.resize(image, (80, 60), interpolation=cv2.INTER_LINEAR)

	ress.append(image)
	histxs.append(histx)

	resizeds.append(resized)
	#for x in range(0, 6):
	#	for i in ress:
	#		filename = sprintf(file,'trial%i.png',x)
	#		cv2.imwrite(file, i)

	for i, pic in enumerate(resizeds):

			saveTo = "/Users/ryokon/Desktop/ryokonTool/tests/" + str(i) + ".png"
			cv2.imwrite(saveTo, pic)



##########################
#training datasets

for imagePath in glob.glob(args["dataset"] + "/*.png"):
	# extract the image filename (assumed to be unique) and
	# load the image, updating the images dictionary
	filename = imagePath[imagePath.rfind("/") + 1:]
	img_rgb = cv2.imread(imagePath)

	hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

	lower_green = np.array([244,244,244])
	upper_green = np.array([255,255,255])

	mask_inverse = cv2.inRange(hsv, lower_green, upper_green)
	mask = cv2.bitwise_not(mask_inverse)

	roi = cv2.bitwise_and(img_rgb,img_rgb, mask = mask)

	image = roi #cv2.resize(roi, (80, 60), interpolation=cv2.INTER_LINEAR)


	images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# extract a 3D RGB color histogram from the image,
	# using 8 bins per channel, normalize, and update
	# the index
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist).flatten()
	index[filename] = hist


##############################

tests1 = []
tests2 = []
tests3 = []
tests4 = []
tests5 = []
tests6 = []



results1 = {}
results2 = {}
results3 = {}
results4 = {}
results5 = {}
results6 = {}


for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[0], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results1 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[1], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results2 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[2], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results3 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[3], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results4 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[4], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results5 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (k, hist) in index.items():
	d = cv2.compareHist(hist, histxs[5], cv2.cv.CV_COMP_BHATTACHARYYA)
	results[k] = d
results6 = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)

for (v, k) in results1[0:10]:
	#print os.path.basename(k)
	tests1.append(os.path.basename(k))

for (v, k) in results2[0:10]:
	#print os.path.basename(k)
	tests2.append(os.path.basename(k))

for (v, k) in results3[0:10]:
	#print os.path.basename(k)
	tests3.append(os.path.basename(k))

for (v, k) in results4[0:10]:
	#print os.path.basename(k)
	tests4.append(os.path.basename(k))

for (v, k) in results5[0:10]:
	#print os.path.basename(k)
	tests5.append(os.path.basename(k))

for (v, k) in results6[0:10]:
	#print os.path.basename(k)
	tests6.append(os.path.basename(k))



################################


#print(bigtests[0][1])
#image_pil = Image.open('/Users/ryokon/Desktop/tester6/'+bigtests[0][1]).convert('L')
#image = np.array(image_pil, 'uint8')


################################

recognizer1 = cv2.createLBPHFaceRecognizer()
recognizer2 = cv2.createLBPHFaceRecognizer()
recognizer3 = cv2.createLBPHFaceRecognizer()
recognizer4 = cv2.createLBPHFaceRecognizer()
recognizer5 = cv2.createLBPHFaceRecognizer()
recognizer6 = cv2.createLBPHFaceRecognizer()


################################


#setup for the training images
def get_images_and_labels():

	images1 = []
	images2 = []
	images3 = []
	images4 = []
	images5 = []
	images6 = []

	labels1 = []
	labels2 = []
	labels3 = []
	labels4 = []
	labels5 = []
	labels6 = []


	for i in tests1:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images1.append(image)
		labels1.append(int(i[0:3]))

	for i in tests2:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images2.append(image)
		labels2.append(int(i[0:3]))

	for i in tests3:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images3.append(image)
		labels3.append(int(i[0:3]))

	for i in tests4:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images4.append(image)
		labels4.append(int(i[0:3]))

	for i in tests5:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images5.append(image)
		labels5.append(int(i[0:3]))

	for i in tests6:

		image_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(i)).convert('L')
		image = np.array(image_pil, 'uint8')
		images6.append(image)
		labels6.append(int(i[0:3]))

	return images1, labels1, images2, labels2, images3, labels3, images4, labels4, images5, labels5, images6, labels6


# トレーニング画像を取得
images1, labels1, images2, labels2, images3, labels3, images4, labels4, images5, labels5, images6, labels6 = get_images_and_labels()

# トレーニング実施
recognizer1.train(images1, np.array(labels1))
recognizer2.train(images2, np.array(labels2))
recognizer3.train(images3, np.array(labels3))
recognizer4.train(images4, np.array(labels4))
recognizer5.train(images5, np.array(labels5))
recognizer6.train(images6, np.array(labels6))

#setup for the testing images
def get_images_and_labels2():

	imagesR = []
	labelsR = []

	for x in range(0, 6):

		image_pilR = Image.open('/Users/ryokon/Desktop/ryokonTool/tests/'+str(x)+'.png').convert('L')
		imageR = np.array(image_pilR, 'uint8')
		imagesR.append(imageR)
		labelsR.append(x)

	return imagesR, labelsR

# テスト画像を取得
imagesR, labelsR = get_images_and_labels2()



label1, confidence1 = recognizer1.predict(imagesR[0])
label2, confidence2 = recognizer2.predict(imagesR[1])
label3, confidence3 = recognizer3.predict(imagesR[2])
label4, confidence4 = recognizer4.predict(imagesR[3])
label5, confidence5 = recognizer5.predict(imagesR[4])
label6, confidence6 = recognizer6.predict(imagesR[5])

Final = []

Final.append(label1)
Final.append(label2)
Final.append(label3)
Final.append(label4)
Final.append(label5)
Final.append(label6)

print("Predicted Label: {}, Confidence: {}".format(label1, confidence1))
print("Predicted Label: {}, Confidence: {}".format(label2, confidence2))
print("Predicted Label: {}, Confidence: {}".format(label3, confidence3))
print("Predicted Label: {}, Confidence: {}".format(label4, confidence4))
print("Predicted Label: {}, Confidence: {}".format(label5, confidence5))
print("Predicted Label: {}, Confidence: {}".format(label6, confidence6))


##################################

point = []

monNames = []
monDex = []

columns = defaultdict(list)
with open('pokemonDetailEng.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)


name = columns[1]
dex = columns[2]




a = dex.index(str(label1))
b = dex.index(str(label2))
c = dex.index(str(label3))
d = dex.index(str(label4))
e = dex.index(str(label5))
f = dex.index(str(label6))

point.append(a)
point.append(b)
point.append(c)
point.append(d)
point.append(e)
point.append(f)


n1 = name[a]
n2 = name[b]
n3 = name[c]
n4 = name[d]
n5 = name[e]
n6 = name[f]

monNames.append(n1)
monNames.append(n2)
monNames.append(n3)
monNames.append(n4)
monNames.append(n5)
monNames.append(n6)

for mon in monNames:
	print (mon + "\n")


d1 = dex[a]
d2 = dex[b]
d3 = dex[c]
d4 = dex[d]
d5 = dex[e]
d6 = dex[f]

monDex.append(d1)
monDex.append(d2)
monDex.append(d3)
monDex.append(d4)
monDex.append(d5)
monDex.append(d6)

###################################
point2 = []

columns = defaultdict(list)
with open('pokemonDetail.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        for (i,v) in enumerate(row):
            columns[i].append(v)

dex2 = columns[1]
hp = columns[2]
att = columns[3]
defe = columns[4]
spatt = columns[5]
spdefe = columns[6]
spe = columns[7]

a2 = dex2.index(str(label1))
b2 = dex2.index(str(label2))
c2 = dex2.index(str(label3))
d2 = dex2.index(str(label4))
e2 = dex2.index(str(label5))
f2 = dex2.index(str(label6))

point2.append(a2)
point2.append(b2)
point2.append(c2)
point2.append(d2)
point2.append(e2)
point2.append(f2)
###################################

f = open('ryokonTool.csv', 'a')

writer = csv.writer(f, lineterminator='\n')
writer.writerow(monNames)


f.close()


#######################################

count=0
for i in point:
	dexNo = dex[i]
	mon = name[i]

	g = point2[count]

	h = hp[g]
	a = att[g]
	b = defe[g]
	c = spatt[g]
	d = spdefe[g]
	s = spe[g]

	result_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/tester6/'+str(dexNo)+".png")
	#result_pil = Image.open('/Users/ryokon/Desktop/ryokonTool/outputs/'+str(mon)+".png").convert('L')
	result = np.array(result_pil, 'uint8')


	fig = plt.figure("results")

	ax = fig.add_subplot(1, 6, count+1)
	ax.set_title(mon)
	plt.text(0, 80, "H:" + " " + h)
	plt.text(0, 100, "A:" + " " + a)
	plt.text(0, 120, "B:" + " " + b)
	plt.text(0, 140, "C:" + " " + c)
	plt.text(0, 160, "D:" + " " + d)
	plt.text(0, 180, "S:" + " " + s)
	plt.imshow(result)
	plt.axis("off")
	count +=1


#plt.show()
plt.show(block=False)
input("Hit Enter To Close")
plt.close()

##################################

#cv2.imshow("Show",resizeds[0])
#cv2.imshow("Show",bounding_img)

cv2.waitKey(8000)
cv2.destroyAllWindows()
