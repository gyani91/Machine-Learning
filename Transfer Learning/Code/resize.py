import os, glob
from PIL import Image

size = (299, 299)

def resize_single_file(input_file):
    output_file = os.path.splitext(input_file)[0] + "_resized.jpg"
    if input_file != output_file:
        try:
            img = Image.open(input_file)
            img.thumbnail(size, Image.ANTIALIAS)
            previous_image_size = img.size
            img_new = Image.new("RGB", size)
            img_new.paste(img, (int((size[0] - previous_image_size[0]) / 2),
                               int((size[1] - previous_image_size[1]) / 2)))
            img_new.save(output_file)
            print('Saved : ' + output_file)
            os.remove(input_file)
            return output_file
        except IOError:
            print('Cannot resize: ' + input_file)

def resize(path):
    output_path=path
    for file in glob.glob(path):
        output_path = resize_single_file(file)
    print('Resizing done')
    return output_path