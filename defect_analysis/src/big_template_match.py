import cv2
import numpy as np
import sys

#load image into variable
img_rgb = cv2.imread('../data/surface_tile_for_template_search.png')
#load template
template = cv2.imread('../data/defect_image_crop.jpeg')

#template = cv2.resize(template,(224,224))

#read height and width of template image
w, h = template.shape[0], template.shape[1]

w2,h2 = img_rgb.shape[0],img_rgb.shape[1]
print(w,h)
print(w2,h2)
res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

print(max_loc)


top_left = max_loc 

bottom_right = (top_left[0]+w,top_left[1]+h)

print(top_left)
print(bottom_right)

crop00 = top_left[0]
crop01 = bottom_right[0]
crop10 = top_left[1]
crop11 = bottom_right[1]


#cv2.rectangle(img_rgb,top_left,bottom_right,(0,255,0),2)

#loc = np.where( res >= threshold)
#print(len(loc))
#for pt in zip(*loc[::-1]):
   # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#img_rgb = cv2.resize(img_rgb,(800,600))

cv2.imwrite('../data/matched_location_crop.png',img_rgb[crop10:crop11,crop00:crop01])


#weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2

#model = torchvision.models.resnet50(weights=weights)

#model_ft = torchvision.models.resnet50(pretrained=True)
#feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
#feature_extractor.eval()

#region = img_rgb[crop10:crop11,crop00:crop01]

#print(region.shape)
#preprocess = weights.transforms()

#cropped_region_from_gt = region[875:925,875:925]
##cropped_region_from_template = template[875:925,875:925]

#cv2.imshow('gt',cropped_region_from_gt)
#cv2.imshow('template',cropped_region_from_template)
#cv2.imshow('absdifference',cv2.absdiff(cropped_region_from_gt,cropped_region_from_template))
#cv2.waitKey(0)
#
#dd = cv2.absdiff(cropped_region_from_gt,cropped_region_from_template)

#print(dd.sum())


#img_gt = cv2.cvtColor(cropped_region_from_gt,cv2.COLOR_BGR2RGB)
#im_gt_pil = Image.fromarray(img_gt)

#img_temp = cv2.cvtColor(cropped_region_from_template,cv2.COLOR_BGR2RGB)
#im_temp_pil = Image.fromarray(img_temp)

#model = SentenceTransformer('clip-ViT-B-32')
#images = [im_gt_pil,im_temp_pil]

#encoded_images = model.encode(images)
#
#processed_images = util.paraphrase_mining_embeddings(encoded_images)

#print(processed_images)


#im_gt_tensor = preprocess(im_gt_pil).unsqueeze(0)
#im_temp_tensor = preprocess(im_temp_pil).unsqueeze(0)

#with torch.no_grad():
#	t1 = feature_extractor(im_gt_tensor)
#	t2 = feature_extractor(im_temp_tensor)

#cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
#output = cos(t1,t2)

#print(output)





