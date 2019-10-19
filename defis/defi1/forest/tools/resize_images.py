import cv2
import os

def resize_image(file_name, resolutions=(224,224)):
    """ Resize one image to a new resolution and save to the same file. 
    :param file_name: the name of the image
    :resolutions the new resolution of the images (default:(224, 224))
    """
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, resolutions, interpolation=cv2.INTER_AREA)
    cv2.imwrite(file_name, resized)    

def scan_and_resize(from_dir, resolutions=(224, 224)):
    """ Resize all images of a directories. 
    :param from_dir: the directory with images
    :resolutions the new resolution of the images (default:(224, 224))
    """
    
    for root, dirs, files in os.walk(from_dir):
        for file in files:
            if file[-4:] in ['.jpg', '.png', 'jpeg']:
                 resize_image(os.path.join(root, file), resolutions=resolutions)    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Resize recursively all images of a directory and save into the same place.')
    parser.add_argument('from_dir', type=str, help='the existing directory with images (sub-directories)')
    parser.add_argument('--resolutionx', default=224, type=int, help='the x resolution of the image (default:224)')
    parser.add_argument('--resolutiony', default=224, type=int, help='the y resolution of the image (default:224)')

    args = parser.parse_args()
    
    scan_and_resize(args.from_dir, (args.resolutionx, args.resolutiony))