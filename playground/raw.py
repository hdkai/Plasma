from PIL import Image
from numpy import allclose, argmax, array, histogram, power, uint8
from rawpy import imread, DemosaicAlgorithm, HighlightMode, Params

#with imread("4.CR2") as raw:
with imread("/Users/yusuf/Desktop/fails/raw/3.arw") as raw:
    black_level = array(raw.black_level_per_channel)
    white_level = array(raw.linear_max)
    white_level = raw.white_level if allclose(white_level, 0) else white_level
    saturation_point = (white_level - black_level).min()
    params = Params(
        demosaic_algorithm=DemosaicAlgorithm.AHD,
        use_camera_wb=True,
        no_auto_bright=True,
        user_sat=saturation_point,
        output_bps=8,
        highlight_mode=HighlightMode.Clip,
        gamma=(1, 1)
    )
    rgb = raw.postprocess(params=params)
    rgb = power(rgb / 255., 1. / 2.2)
    rgb = (rgb * 255.).astype(uint8)
    exposure = Image.fromarray(rgb)
    exposure.save("exposure.jpg")