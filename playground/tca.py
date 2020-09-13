# 
#   Rio
#   Copyright (c) 2020 Homedeck, LLC.
#

from json import dumps
from PIL import Image
from requests import get, put
from torch import cat, enable_grad, linspace, meshgrid, stack, tensor, Tensor
from torch.nn import L1Loss, Parameter
from torch.nn.functional import grid_sample
from torch.optim import SGD
from torchvision.transforms import ToPILImage, ToTensor
from typing import List
from urllib.parse import quote

from .device import get_device

def tca_correction (*images: Image.Image) -> Image.Image:
    """
    Appply transverse chromatic aberration correction on an image.

    Parameters:
        images (PIL.Image | list): Input image(s).
        coefficients (Tensor): Cubic red-blue TCA coefficients with shape (2,4). If `None`, it will be computed (can be slow).

    Returns:
        PIL.Image | list: Corrected image(s).
    """
    # Check images
    if len(images) == 0:
        return []
    # Fetch coefficients
    coefficients = _fetch_coefficients(images)
    if coefficients is None:
        return images if len(images) > 1 else images[0]
    # Save EXIF
    exifs = [image.info.get("exif") for image in images]
    # Create exposure stack tensor
    device = get_device()
    image_tensors = [ToTensor()(image) for image in images]
    exposure_stack = stack(image_tensors, dim=0).to(device)
    coefficients = coefficients.to(device)
    # Apply
    result_stack = _tca_forward(exposure_stack, coefficients)
    # Convert back to image
    exposures = result_stack.split(1, dim=0)
    images = [ToPILImage()(exposure.squeeze(dim=0).cpu()) for exposure in exposures]
    # Add EXIF and return
    for image, exif in zip(images, exifs):
        image.info["exif"] = exif
    return images if len(images) > 1 else images[0]

def _fetch_coefficients (images: List[Image.Image]) -> Tensor:
    """
    Fetch coefficients for TCA correction.
    If coefficients are not available, they will be computed.

    Parameters:
        images (list): Input images.

    Returns:
        Tensor: Quadratic red-blue TCA coefficients with shape (2,3).
    """
    # Get EXIF metadata
    CAMERA_MAKER_EXIF_TAG = 271
    LENS_MAKER_EXIF_TAG = 42035
    LENS_MODEL_EXIF_TAG = 42036
    FOCAL_LENGTH_EXIF_TAG = 37386
    metadata = images[0].getexif()
    if metadata is None:
        print("Rio Error: Image does not have EXIF metadata for TCA correction")
        return None
    camera_maker = metadata.get(CAMERA_MAKER_EXIF_TAG)
    lens_maker = metadata.get(LENS_MAKER_EXIF_TAG) or camera_maker
    lens_model = metadata.get(LENS_MODEL_EXIF_TAG)
    focal_length = metadata.get(FOCAL_LENGTH_EXIF_TAG)
    if lens_maker is None or lens_model is None or focal_length is None:
        print("Rio Error: Image does not have lens metadata for TCA correction")
        return None
    # Fetch coefficients
    lens_maker = quote(lens_maker.replace(".", "_").replace("/", "_"))
    lens_model = quote(lens_model.replace(".", "_").replace("/", "_"))
    focal_length = focal_length[0] / focal_length[1] if isinstance(focal_length, tuple) else focal_length
    focal_length = f"{float(focal_length):.3f}".replace(".", "_")
    uri = f"https://homedeck-rio.firebaseio.com/tca/{lens_maker}/{lens_model}/{focal_length}.json"
    response = get(uri)
    coefficients = response.json()
    KEYS = ["ra", "rb", "rc", "ba", "bb", "bc"]
    if response.status_code == 200 and coefficients is not None:
        ra, rb, rc, ba, bb, bc = [float(coefficients[key]) for key in KEYS]
        coefficients = tensor([
            [ra, rb, rc],
            [ba, bb, bc]
        ])
        return coefficients
    # Compute coefficients
    coefficients = _compute_coefficients(images)
    if coefficients is None:
        return None
    # Upload coefficients
    payload = { k: v for k, v in zip(KEYS, coefficients.flatten().tolist()) }
    response = put(uri, data=dumps(payload))
    # Return
    return coefficients

def _compute_coefficients (images: List[Image.Image]) -> Tensor: # INCOMPLETE
    """
    Compute cubic transverse chromatic aberration correction coefficients.
    Once computed, these coefficients can be applied to all images captured by the same camera and lens pair.
    If the coefficients cannot be computed, `None` is returned.

    Parameters:
        images (list): Input images.
    
    Returns:
        Tensor: Quadratic red-blue TCA coefficients with shape (2,3).
    """
    # Get objective
    objective = _tca_objective(images)
    if objective is None:
        return None
    device = get_device()
    objective = objective.to(device)
    # Define coeffs # Use quadratic distortion model as in lensfun
    coeffs = Parameter(tensor([
        [0., 0., 1.], # red
        [0., 0., 1.]  # blue
    ], device=device, requires_grad=True))
    l1_loss = L1Loss().to(device)
    optimizer = SGD([coeffs], lr=5e-5)
    # Optimize
    best_coeffs = None
    best_loss = None
    with enable_grad(): # In case caller has disabled grad
        for _ in range(10):
            optimizer.zero_grad()
            prediction = _tca_forward(objective, coeffs)
            r, g, b = prediction.split(1, dim=1)
            loss = l1_loss(r, g) + l1_loss(b, g)
            if best_loss is None or loss < best_loss:
                best_coeffs = coeffs.detach().clone()
                best_loss = loss.detach().clone()
            loss.backward()
            optimizer.step()
    # Return coefficients
    return best_coeffs.data.cpu()

def _tca_forward (input: Tensor, coefficients: Tensor):
    """
    Compute the quadratic radial lens distortion forward pass.

    Parameters:
        input (Tensor): Image tensor with shape (N,3,H,W).
        coefficients (Tensor): Quadratic red-blue TCA coefficients with shape (2,3).

    Returns:
        Tensor: Transformed image tensor with shape (N,3,H,W).
    """
    # Construct radial sampling field
    batch, _, height, width = input.shape
    hg, wg = meshgrid(linspace(-1., 1., height), linspace(-1., 1., width))
    hg = hg.repeat(batch, 1, 1).unsqueeze(dim=3).to(input.device)
    wg = wg.repeat(batch, 1, 1).unsqueeze(dim=3).to(input.device)
    sample_field = cat([wg, hg], dim=3)
    r_dst = sample_field.norm(dim=3, keepdim=True)
    # Compute distortions
    red_distortion = coefficients[0,0] * r_dst.pow(2) + coefficients[0,1] * r_dst.pow(1) + coefficients[0,2]
    blue_distortion = coefficients[1,0] * r_dst.pow(2) + coefficients[1,1] * r_dst.pow(1) + coefficients[1,2]
    # Compute sample grids
    red_grid = sample_field * red_distortion
    blue_grid = sample_field * blue_distortion
    # Sample
    red, green, blue = input.split(1, dim=1)
    red_shifted = grid_sample(red, red_grid, mode="bilinear", padding_mode="border", align_corners=False)
    blue_shifted = grid_sample(blue, blue_grid, mode="bilinear", padding_mode="border", align_corners=False)
    # Combine
    result = cat([red_shifted, green, blue_shifted], dim=1)
    return result

def _tca_objective (images: List[Image.Image]) -> Tensor:
    """
    Select an ideal objective image for the TCA optimization.
    This function requires the exposure bias value on all exposures.

    Parameters:
        images (list): Input images.

    Returns:
        Tensor: Objective image.
    """
    # Trivial case
    if len(images) == 1:
        image = images[0]
        return ToTensor()(image).unsqueeze(dim=0)
    # Get EXIF
    EXPOSURE_BIAS_EXIF_TAG = 37380
    bias_values = [image.getexif().get(EXPOSURE_BIAS_EXIF_TAG) for image in images]
    # Check
    if any([x is None for x in bias_values]):
        print("Rio Error: Images lack exposure bias for TCA objective selection")
        return None
    # Get values
    bias_values = [x[0] / x[1] if isinstance(x, tuple) else x for x in bias_values]
    abs_bias_values = [abs(float(x)) for x in bias_values]
    # Get lowest absolute
    middle_exposure, _ = next(iter(sorted(zip(images, abs_bias_values), key=lambda x: x[1])))
    return ToTensor()(middle_exposure).unsqueeze(dim=0)