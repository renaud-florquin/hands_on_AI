import os
import shutil
from sklearn.model_selection import train_test_split

def split_images(from_dir, to_dir, validation_percent=0.20, test_percent=0.10):
    """ Based on an existing directory (from_dir) with sub-directories containing the different 
    categories of images, the function create a new structure of directories by splitting the images 
    in 3 sets: train, validation et test. 
    Remark: it's a copy of images (not a move)

    :param from_dir: the existing directory with categories (sub-directories) and images
    :param to_dir: the new directory to create with the 3 sets
    :param validation_percent: the validation percentage between [0..1] (default:0.2)
    :param test_percent: the test percentage between [0..1] (default:0.1)
    """
    
    phases = ['train', 'validation', 'test']
    # scan the from_dir
    x = []
    y = []
    categories = [d for d in os.listdir(from_dir) if os.path.isdir(os.path.join(from_dir, d))]
    for category in categories:
        cat_x = [f for f in os.listdir(os.path.join(from_dir, category))]
        cat_y = [category for i in range(len(cat_x))]
        x += cat_x
        y += cat_y
    assert len(x) == len(y)

    # split images to three sets
    x_train, x_other, y_train, y_other = train_test_split(x, y, test_size=(validation_percent+test_percent))
    percent_test_validation = test_percent / (validation_percent + test_percent)
    x_validation, x_test, y_validation, y_test = train_test_split(x_other, y_other, test_size=percent_test_validation)
    
    # create the directories structure and copy the files
    os.makedirs(to_dir)
    for x, y, category in zip([x_train, x_validation, x_test], [y_train, y_validation, y_test], phases):
        phase_dir = os.path.join(to_dir, category)        
        os.makedirs(phase_dir)
        for category in categories:
            from_dir_cat = os.path.join(from_dir, category)
            to_dir_cat = os.path.join(phase_dir, category)
            os.makedirs(to_dir_cat)
            files = [f for f, cat in zip(x, y) if cat == category]
            for f in files:
                shutil.copy(os.path.join(from_dir_cat, f), os.path.join(to_dir_cat, f))    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Split a set of classified images into 3 set: train/validation/test.')
    parser.add_argument('from_dir', type=str, help='the existing directory with categories (sub-directories)')
    parser.add_argument('to_dir', type=str, help='the destination directory to create')
    parser.add_argument('--validation_percent', default=0.2, type=float, help=' the validation percentage between [0..1] (default:0.2)')
    parser.add_argument('--test_percent', default=0.1, type=float, help=' the test percentage between [0..1] (default:0.1)')

    args = parser.parse_args()
    
    split_images(args.from_dir, args.to_dir, args.validation_percent, args.test_percent)