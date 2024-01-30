'''
Author:
    Thomas Agervig Jensen 
    agervig@outlook.com

Inspired by:
    https://learnopencv.com/selective-search-for-object-detection-cpp-python/

Usage:
    Search input image for region proposal default is 100 regions.
    press "m" while running the program for more regions and "l" for less regions

Arguments:
    argv[1] = input image
    argv[2] = selective search method: f = fast, q = quality
'''

import sys
import cv2





if __name__ == "__main__": 
    if len(sys.argv) < 3:
        print("ERROR! READ DOCUMENTATION TO CORRECT INPUTS")
        print(__doc__)
        sys.exit(1) 

    #command line arguments
    input_img = sys.argv[1]
    ss_mode = sys.argv[2]

    # Speed up using multithreads
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)


    # Read image and resize height to 200
    img = cv2.imread(input_img)

    new_height = 600
    original_height = img.shape[0]
    original_width = img.shape[1]
    new_width = int(original_width * new_height / original_height)
    img = cv2.resize(img, (new_width, new_height))


    # Create Selective Search Segmentation Ojbect using default parameters 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)

    # Choose between fast and slow selective search method (slow has high recall, fast low recall)
    if ss_mode == "f":
        ss.switchToSelectiveSearchFast()
    # Run selective search on input img
    elif ss_mode == "q":
        ss.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)
    
    # Run selective search segmentation on input image
    regions = ss.process()
    print(f"Total number of Region Proposals:{len(regions)}")

    # Define number of region proposals to show
    num_regions_showed = 100
    increment = 50

    while True:
        img_out = img.copy()
        # Iterate over all the region proposals
        for i, reg in enumerate(regions):
            # Draw rectangle for region proposal till num_regions_showed
            if (i < num_regions_showed):
                x, y, w, h = reg
                cv2.rectangle(img_out, (x, y) , (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # show output
        cv2.imshow("Selective Search output", img_out)

        key_press = cv2.waitKey(0) & 0xFF
        # m is pressed:
        if key_press == 109:
            num_regions_showed += increment
        # l is pressed:
        if key_press == 108:
            num_regions_showed -= increment
        # q is pressed:
        if key_press == 113:
            break

    cv2.destroyAllWindows()