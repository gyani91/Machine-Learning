from PIL import Image
import glob
import os,sys

def convert_image(input_file):
    try:
        image = Image.open(input_file)
    except IOError:
        print("Cant load: ", input_file)
        sys.exit(1)
    try:
        output_image = Image.new("RGB", image.size)
        output_image.paste(image)
        output_file = os.path.splitext(input_file)[0] + '_converted.jpg'
        output_image.save(output_file)
        print('Saved : ' + output_file)
        os.remove(input_file)
        return output_file
    except IOError:
        print('Cannot convert: ' + input_file)

def convert(path):
    output_path=path
    for file in glob.glob(path):
        if os.path.splitext(file)[1] != '.jpg':
            output_path = convert_image(file)
    print('Images converted')
    return output_path