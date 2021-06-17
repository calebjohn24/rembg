import functools
import io

import numpy as np
from PIL import Image
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage.morphology import binary_erosion
from PIL import Image, ImageEnhance


from .u2net import detect


def alpha_matting_cutout(
    img,
    mask,
    foreground_threshold,
    background_threshold,
    erode_structure_size,
    base_size,
):
    size = img.size

    img.thumbnail((base_size, base_size), Image.LANCZOS)
    mask = mask.resize(img.size, Image.LANCZOS)

    img = np.copy(np.asarray(img))
    mask = np.copy(np.asarray(mask))

    

    super_threshold_indices = mask < 50
    mask[super_threshold_indices] = 0

    super_threshold_indices = mask > 240
    mask[super_threshold_indices] = 255

    mask_new = Image.fromarray(mask)
    mask_new.save("mask.png")

    # guess likely foreground/background
    is_foreground = mask > 240
    is_background = mask < 50

    erode_structure_size = 3
    # erode foreground/background
    structure = None
    if erode_structure_size > 0:
        structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.int)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # build trimap
    # 0   = background
    # 128 = unknown
    # 255 = foreground
    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)

    trimap[is_foreground] = 255
    trimap[is_background] = 0

    result_arr = np.dstack((img, trimap))
    result = Image.fromarray(result_arr)
    result.save("nomatting.png")


    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=200)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    trimapImg = Image.fromarray(np.copy(trimap))
    trimapImg.save("trimap.png")
    # build the cutout image
    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha, gradient_weight=0.005)
    cutout = stack_images(foreground, trimap)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)

    
    opacity = cutout[:, :, 3]
    super_threshold_indices_low = cutout[:, :, 3] <= 20
    super_threshold_indices_high = cutout[:, :, 3] >= 230
    opacity[super_threshold_indices_low] = 0
    opacity[super_threshold_indices_high] = 255

    mid = (cutout[:, :, 3] > 180) & (cutout[:, :, 3] < 230)
    opacity[mid] += 25

    mid = (cutout[:, :, 3] > 100) & (cutout[:, :, 3] <= 180)
    opacity[mid] += 40

    mid = (cutout[:, :, 3] > 50) & (cutout[:, :, 3] <= 100)
    opacity[mid] += 60

    mid = (cutout[:, :, 3] > 20) & (cutout[:, :, 3] <= 50)
    opacity[mid] += 110

    cutout = Image.fromarray(cutout)
    cutoutImg = Image.fromarray(np.copy(cutout))
    cutoutImg.save("cutout.png")
    cutout = cutout.resize(size, Image.LANCZOS)

    return cutout


def naive_cutout(img, mask):
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))
    return cutout


@functools.lru_cache(maxsize=None)
def get_model(model_name):
    if model_name == "u2netp":
        return detect.load_model(model_name="u2netp")
    if model_name == "u2net_human_seg":
        return detect.load_model(model_name="u2net_human_seg")
    else:
        return detect.load_model(model_name="u2net")


def remove(
    data,
    model_name="u2net",
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=40,
    alpha_matting_erode_structure_size=1,
    alpha_matting_base_size=1000,
):
    model = get_model(model_name)
    img = Image.open(io.BytesIO(data)).convert("RGB")

    enhancer = ImageEnhance.Sharpness(img)

    factor = 2
    imgs = enhancer.enhance(factor)
    imgs.save('sharpened-image.png');

    enhancer_2 = ImageEnhance.Contrast(imgs)


    factor = 1.1 #increase contrast
    im_c = enhancer_2.enhance(factor)
    im_c.save('more-contrast-image.png')

    factor = 1.1
    enhancer_3 = ImageEnhance.Color(im_c)
    im_color = enhancer_3.enhance(factor)
    im_color.save('more-saturation-image.png')

    factor = 0.9
    enhancer_4 = ImageEnhance.Brightness(im_color)
    im_output = enhancer_4.enhance(factor)
    im_output.save('brightness-image.png')

    mask = detect.predict(model, np.array(im_output)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")

    return bio.getbuffer()
