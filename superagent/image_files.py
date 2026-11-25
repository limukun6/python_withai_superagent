"""Decode single-frame bitmap inputs using a versioned, deterministic pixel contract."""
import io
from pathlib import PurePosixPath

import numpy as np

LEGACY_POLICY='png_jpeg_exif_white_background_bilinear_v1'
DECODER_POLICY='single_frame_bitmap_exif_white_background_bilinear_v2'
LEGACY_FORMATS={'.png':'PNG','.jpg':'JPEG','.jpeg':'JPEG'}
BITMAP_FORMATS={**LEGACY_FORMATS,'.bmp':'BMP','.tif':'TIFF','.tiff':'TIFF','.webp':'WEBP','.gif':'GIF'}
MAX_SOURCE_PIXELS=16_777_216
MAX_FILE_BYTES=64*1024*1024
MAX_OUTPUT_BYTES=256*1024*1024


def image_contract(value):
    if not isinstance(value,dict) or set(value)!={'width','height','mode','resize'}:
        raise ValueError('Declare image_preprocessing with width, height, mode, and resize')
    if any(type(value[k]) is not int or not 1<=value[k]<=512 for k in ('width','height')):
        raise ValueError('Declared image width and height must be integers in [1,512]')
    if value['mode'] not in ('RGB','L') or value['resize']!='stretch_bilinear':
        raise ValueError('Image files require mode RGB/L and resize=stretch_bilinear')
    return dict(value)


def load_image_rows(archive,manifest,split,paths):
    from PIL import Image,ImageOps
    from .schema import _member
    contract=image_contract(manifest.get('image_preprocessing'))
    channels=3 if contract['mode']=='RGB' else 1
    if len(paths)*contract['width']*contract['height']*channels>MAX_OUTPUT_BYTES:
        raise ValueError('Decoded image split exceeds the 256 MiB in-memory input limit')
    if not paths:raise ValueError('Empty image split')
    values=[]
    formats = LEGACY_FORMATS if manifest.get('_image_decoder_policy') == LEGACY_POLICY else BITMAP_FORMATS
    for raw in paths:
        descriptor={**manifest,'splits':{split:{'path':raw}}}
        _,member=_member(archive,descriptor,split)
        suffix=PurePosixPath(member).suffix.casefold()
        expected=formats.get(suffix)
        if expected is None:raise ValueError('Image path has an unsupported bitmap extension')
        if archive.getinfo(member).file_size>MAX_FILE_BYTES:raise ValueError('Encoded image exceeds the 64 MiB per-file limit')
        with Image.open(io.BytesIO(archive.read(member)),formats=list(dict.fromkeys(formats.values()))) as source:
            if source.format!=expected:raise ValueError('Image content format does not match its extension')
            if source.width*source.height>MAX_SOURCE_PIXELS:raise ValueError('Source image exceeds the 16-megapixel decoder limit')
            if getattr(source,'n_frames',1)!=1:raise ValueError('Animated or multiframe images are not supported')
            if source.mode not in ('1','L','LA','P','RGB','RGBA','CMYK'):
                raise ValueError('Image files require supported 8-bit color or grayscale pixels')
            source.load()
            rgba=ImageOps.exif_transpose(source).convert('RGBA')
            background=Image.new('RGBA',rgba.size,(255,255,255,255))
            image=Image.alpha_composite(background,rgba).convert(contract['mode'])
            image=image.resize((contract['width'],contract['height']),resample=Image.Resampling.BILINEAR)
            values.append(np.asarray(image,dtype=np.uint8).copy())
    return np.stack(values),contract
