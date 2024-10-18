import cv2
import os

# Read image (using raw string for the file path)
img1_path = r'.\Images\img1.jpg'
img2_path = r'.\Images\img2.jpg'

# Read the images
img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

# if img1 is None or img2 is None:
#     print("Error: Failed to load one or both images.")
# else:
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()

#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1,des2,k=2)

#     # Applying ratio test to threshold the images
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good.append([m])

#     img3 = cv2.drawMatchesKnn(
#         img1, kp1, img2, kp2, good, None, 
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#       # Resize if the image is too large
#     max_height = 500
#     height, width = img3.shape[:2]
#     if height > max_height:
#         scale = max_height / height
#         img3 = cv2.resize(img3, (int(width * scale), max_height))


#     # Display the result
#     cv2.imshow('SIFT Matches', img3)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



#Brute forcing with binary descriptors
if img1 is None or img2 is None:
    print("Error: Failed to load one or both images.")
else:
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    n = min(50, len(matches))
    img3 = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

        # Resize if the image is too large
    max_height = 500
    height, width = img3.shape[:2]
    if height > max_height:
        scale = max_height / height
        img3 = cv2.resize(img3, (int(width * scale), max_height))

    cv2.imshow('Binary descriptor SIFT',img3)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


