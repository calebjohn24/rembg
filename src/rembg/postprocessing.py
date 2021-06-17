from PIL import Image
import cv2
import skimage
import numpy as np

def extact_alpha_channel(image):
    """
    Extracts alpha channel from RGBA image
    :param image: RGBA pil image
    :return: RGB Pil image
    """
    # Extract just the alpha channel
    alpha = image.split()[-1]
    # Create a new image with an opaque black background
    bg = Image.new("RGBA", image.size, (0, 0, 0, 255))
    # Copy the alpha channel to the new image using itself as the mask
    bg.paste(alpha, mask=alpha)
    return bg.convert("RGB")

def blur_edges( imaged):
    """
    Blurs the edges of the image
    :param imaged: RGBA Pil image
    :return: RGBA PIL  image
    """
    image = np.array(imaged)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    # extract alpha channel
    a = image[:, :, 3]
    # blur alpha channel
    ab = cv2.GaussianBlur(a, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    # stretch so that 255 -> 255 and 127.5 -> 0
    aa = skimage.exposure.rescale_intensity(ab, in_range=(140, 255), out_range=(0, 255))
    # replace alpha channel in input with new alpha channel
    out = image.copy()
    out[:, :, 3] = aa
    image = cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(image)

def remove_too_transparent_borders( mask, tranp_val=31):
    """
    Marks all pixels in the mask with a transparency greater than $tranp_val as opaque.
    Pixels with transparency less than $tranp_val, as fully transparent
    :param tranp_val: Integer value.
    :return: Processed mask
    """
    mask = np.array(mask.convert("L"))
    height, weight = mask.shape
    for h in range(height):
        for w in range(weight):
            val = mask[h, w]
            if val > tranp_val:
                mask[h, w] = 255
            else:
                mask[h, w] = 0
    return Image.fromarray(mask)

def run(image, orig_image):
    """
    Runs an image post-processing algorithm to improve background removal quality.
    :param model: The class of the neural network used to remove the background.
    :param image: Image without background
    :param orig_image: Source image
    """
    mask = remove_too_transparent_borders(extact_alpha_channel(image))
    empty = Image.new("RGBA", orig_image.size)
    image = Image.composite(orig_image, empty, mask)
    image = blur_edges(image)

    mask = remove_too_transparent_borders(extact_alpha_channel(image))
    empty = Image.new("RGBA", orig_image.size)
    image = Image.composite(orig_image, empty, mask)
    image = blur_edges(image)
    return image