"""This module provides a SoftForegroundSegmenter which segments the foreground human subjects from the background.

Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
from onnxruntime import InferenceSession


ONNX_EP = ["CUDAExecutionProvider", "CPUExecutionProvider"]
UINT8_MAX = np.iinfo(np.uint8).max
UINT16_MAX = np.iinfo(np.uint16).max


class ImageFormatError(Exception):
    """Exception raised for invalid image formats."""
    pass


class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""
    pass


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """Preprocesses a BGR image for DNN. Turning to float if not already and normalizing to [0, 1].

    Normalization of uint images is done by dividing by brightest possible value (e.g. 255 for uint8).

    Arguments:
        img: The image to preprocess, can be uint8, uint16, float16, float32 or float64.

    Returns:
        The preprocessed image in np.float32 format.

    Raises:
        ImageFormatError: If the image is not three channels or not uint8, uint16, float16, float32 or float64.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ImageFormatError("image must be 3 channels, got shape: {img.shape}")
    if img.dtype not in [np.uint8, np.uint16, np.float16, np.float32, np.float64]:  # noqa: PLR6201
        raise ImageFormatError("image must be uint8 or float16, float32, float64")

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / UINT8_MAX
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / UINT16_MAX
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


def prepare_image_for_model(image: np.ndarray, roi_size: int = 512) -> tuple[np.ndarray, dict]:
    """Prepare any input image for model inference by resizing to roi_size x roi_size.

    This function takes an image of any size and prepares it for a model that expects
    a square input (e.g., 512x512). It handles aspect ratio preservation by padding
    with replicated border values.

    Args:
        image: Input image of any size
        roi_size: Target size for the model (default 512)

    Returns:
        tuple: (preprocessed_image, metadata_dict)
            - preprocessed_image: Image resized to roi_size x roi_size
            - metadata_dict: Contains information needed to composite back to original size
    """
    # Get original shape
    original_shape = image.shape[:2]  # (height, width)

    # Calculate padding to make the image square
    if original_shape[0] < original_shape[1]:
        pad_h = (original_shape[1] - original_shape[0]) // 2
        pad_w = 0
        pad_h_extra = original_shape[1] - original_shape[0] - pad_h
        pad_w_extra = 0
    elif original_shape[0] > original_shape[1]:
        pad_w = (original_shape[0] - original_shape[1]) // 2
        pad_h = 0
        pad_w_extra = original_shape[0] - original_shape[1] - pad_w
        pad_h_extra = 0
    else:
        pad_h = pad_w = pad_h_extra = pad_w_extra = 0

    # Pad the image to make it square
    padded_image = cv2.copyMakeBorder(
        image,
        top=pad_h,
        bottom=pad_h_extra,
        left=pad_w,
        right=pad_w_extra,
        borderType=cv2.BORDER_REPLICATE,
    )

    square_shape = padded_image.shape[:2]

    while padded_image.shape[1] > roi_size * 3 and padded_image.shape[0] > roi_size * 3:
        padded_image = cv2.pyrDown(padded_image)

    resized_image = cv2.resize(padded_image, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)

    metadata = {
        "original_shape": original_shape,
        "square_shape": square_shape,
        "original_padding": (pad_h, pad_w, pad_h_extra, pad_w_extra),
    }

    return resized_image, metadata


def composite_model_output_to_image(
    model_output: np.ndarray, metadata: dict, interp_mode: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """Composite model output back to the original image size.

    Takes the model output (which should be roi_size x roi_size) and composites it
    back to the original image dimensions using the metadata from prepare_image_for_model.

    Args:
        model_output: Output from the model (roi_size x roi_size)
        metadata: Metadata dict returned from prepare_image_for_model
        interp_mode: Interpolation mode for resizing (default INTER_NEAREST for discrete outputs)

    Returns:
        np.ndarray: Output composited to original image size
    """
    pad_h, pad_w, pad_h_extra, pad_w_extra = metadata["original_padding"]

    # Resize the entire model output back to the square shape
    square_shape = metadata["square_shape"]
    resized_to_square = cv2.resize(model_output, (square_shape[1], square_shape[0]), interpolation=interp_mode)

    # Remove the padding to get back to original dimensions
    if pad_h > 0 or pad_h_extra > 0:
        final_output = resized_to_square[pad_h : square_shape[0] - pad_h_extra, :]
    elif pad_w > 0 or pad_w_extra > 0:
        final_output = resized_to_square[:, pad_w : square_shape[1] - pad_w_extra]
    else:
        final_output = resized_to_square

    return final_output


class RuntimeSession(InferenceSession):
    """The runtime session."""

    def __init__(self, onnx_model: Union[str, Path], providers: Optional[list[str]] = None) -> None:
        """Create a runtime session.

        Args:
            onnx_model: The path to the onnx model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
        """
        super().__init__(str(onnx_model), providers=providers or ONNX_EP)
        self.onnx_model_path: Path = Path(onnx_model)

    @property
    def input_name(self) -> str:
        """Get the name of the input tensor."""
        return self.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        """Run the model on the input tensor."""
        x = x.astype(np.float32)
        return self.run(None, {self.input_name: x})


class PixelwiseEstimator:
    """Given an input image, estimates the pixelwise (dense) output (e.g., normal map, depth map, etc.)."""

    def __init__(self, onnx_model: Union[str, Path], providers: Optional[list[str]] = None):
        """Creates a pixelwise estimator.

        Arguments:
$0
            onnx_model: Path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].

        Raises:
            TypeError: If onnx_model is not a string or Path.
            ModelNotFoundError: If the model file does not exist.
            ModelError: If the provided model has an undeclared or incorrect roi type.
        """
        if not isinstance(onnx_model, (str, Path)):
            raise TypeError(f"onnx_model should be a string or Path, got {type(onnx_model)}")
        onnx_model = Path(onnx_model)
        if not onnx_model.exists():
            raise ModelNotFoundError(f"model {onnx_model} does not exist")

        self.onnx_model = onnx_model

        self.roi_size = 512

        self.onnx_sess = RuntimeSession(str(onnx_model), providers=providers)

    @staticmethod
    def inference(input_img: np.ndarray, onnx_sess: RuntimeSession) -> np.ndarray:
        """Predict the pixelwise (dense) map given an input image.

        Args:
            input_img: Input image.
            onnx_sess: ONNX inference session.

        Returns:
            Predicted output map.
        """
        input_tensor = onnx_sess.get_inputs()[0]
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        input_img = np.transpose(input_img, (2, 0, 1)).reshape(1, *input_shape[1:])  # HWC to BCHW
        pred_onnx = onnx_sess.run(None, {input_name: input_img.astype(np.float32)})

        return pred_onnx

    def _estimate_dense_map(self, image: np.ndarray) -> tuple[np.ndarray]:
        """Estimating dense maps from image input."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image should be a numpy array, got {type(image)}")

        image_bgr = preprocess_img(image)
        processed_image, metadata = prepare_image_for_model(image_bgr, self.roi_size)
        output = self.inference(processed_image, self.onnx_sess)

        return output, metadata


class SoftForegroundSegmenter(PixelwiseEstimator):
    """Estimates the soft foreground segmentation mask of human in an image."""

    def __init__(
        self,
        onnx_model: Union[str, Path],
        providers: Optional[list[str]] = None,
        binarization_threshold: Optional[float] = None,
    ):
        """Creates a soft foreground segmenter to segment the foreground human subjects in an image.

        Arguments:
            onnx_model: A path to an ONNX model.
            providers: Optional list of ONNX execution providers to use, defaults to [GPU, CPU].
            binarization_threshold: Threshold above which the mask is considered foreground. When None, the mask is returned as is.

        Raises:
            TypeError: if onnx_model is not a string or Path.
            ModelNotFoundError: if the model file does not exist.
        """
        super().__init__(
            onnx_model,
            providers=providers,
        )
        self.binarization_threshold = binarization_threshold

    def estimate_foreground_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Predict the soft foreground/background segmentation given input image."""
        mask, metadata = self._estimate_dense_map(image)
        mask = mask[0][0]
        mask = np.transpose(mask, (1, 2, 0))

        # post_process to get the final segmentation mask and composite it onto the original size
        segmented_image = composite_model_output_to_image(mask, metadata, interp_mode=cv2.INTER_CUBIC)

        # clip the mask to [0, 1]
        segmented_image = np.clip(segmented_image, 0, 1)

        # Apply threshold if binarization_threshold is set
        if self.binarization_threshold:
            return ((segmented_image > self.binarization_threshold) * 1).astype(np.uint8)

        return segmented_image
