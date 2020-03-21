from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def decide_to_apply(cond):
    decision = cond and np.random.random(1) > 0.5
    return decision


def random_flip(
    image, 
    target, 
    apply_flip_axis_x, 
    apply_flip_axis_y, 
    apply_flip_axis_z):
    
    apply_flip = [apply_flip_axis_x, apply_flip_axis_y, apply_flip_axis_z]

    for i in range(len(apply_flip)):
        if decide_to_apply(apply_flip[i]):
            print("Applying fip for axis " + str(i))
            image = np.flip(image, axis=i)
            target = np.flip(target, axis=i)
    
    return (image, target)
    

def add_gaussian_offset(image, apply_gaussian_offset, sigma):
    
    if decide_to_apply(apply_gaussian_offset):
        print("Applying gaussian offset")
        offsets = np.random.normal(0, sigma, ([1] * (image.ndim - 1) + [image.shape[-1]]))
        image += offsets
    
    return image


def add_gaussian_noise(image, apply_gaussian_noise, sigma):
 
    if decide_to_apply(apply_gaussian_noise):
        print("Applying noise")
        noise = np.random.normal(0, sigma, image.shape)
        image += noise
        
    return image

def elastic_transform(imgage, apply_elastic_transfor, alpha, sigma):

    assert len(alpha) == len(sigma), "Dimensions of alpha and sigma are different for elastic transform"
    
    if decide_to_apply(apply_elastic_transfor):
        print("Applying elastic transformation")
        channelbool = image.ndim - len(alpha)
        out = np.zeros((len(alpha) + channelbool, ) + image.shape)

        # Generate a Gaussian filter, leaving channel dimensions zeroes
        for jj in range(len(alpha)):
            array = (np.random.rand(*image.shape) * 2 - 1)
            out[jj] = gaussian_filter(
                array, 
                sigma[jj],
                mode="constant", 
                cval=0) * alpha[jj]

        # Map mask to indices
        shapes = list(map(lambda x: slice(0, x, None), image.shape))
        grid = np.broadcast_arrays(*np.ogrid[shapes])
        indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

        # Transform image based on masked indices
        transformed_image = map_coordinates(
            image, 
            indices, 
            order=0,
            mode='reflect').reshape(image.shape)

       
    else:
        transformed_image = image

    return transformed_image
    
    
def apply_transformations(
    image,
    target,
    apply_flip_axis_x = True,
    apply_flip_axis_y = True,
    apply_flip_axis_z = False,
    apply_gaussian_offset = True,
    apply_gaussian_noise = True,
    apply_elastic_transfor = True,
    sigma_gaussian_offset = 0.1,
    sigma_gaussian_noise = 0.05,
    alpha_elastic = [0,0,0],
    sigma_elastic = [1,1,1] 
):
    assert image.shape == target.shape, "Image and target have different shapes"
    
    image = image.astype(float)
    target = target.astype(float)
    image, target = random_flip(image, target, apply_flip_axis_x, apply_flip_axis_y, apply_flip_axis_z)
    image = add_gaussian_offset(image, apply_gaussian_offset, sigma = sigma_gaussian_offset)
    image = add_gaussian_noise(image, apply_gaussian_noise, sigma = sigma_gaussian_noise)
    image = elastic_transform(image, apply_elastic_transfor, alpha = alpha_elastic, sigma = sigma_elastic)
    
    image = image.astype(np.float32)
    target = target.astype(np.float32)
    
    return (image, target)