import cv2
import numpy as np

class TVL1OpticalFlowExtractor:
    """
    OpenCV GPU-accelerated TV-L1 Optical Flow extractor for 5 consecutive frame pairs (K=6 frames -> 5 flows).
    Replicates the standard two-stream HAR baseline from the paper.
    """
    def __init__(self, n_scales: int = 5, warps: int = 5, iterations: int = 30):
        # Check if CUDA is available for OpenCV GPU acceleration
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            self.tvl1 = cv2.cuda_OpticalFlowDual_TVL1.create(
                numScales=n_scales,
                numWarps=warps,
                numIters=iterations
            )
        else:
            try:
                # Fallback to opencv-contrib CPU implementation if CUDA is unavailable
                self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
                    numScales=n_scales,
                    numWarps=warps,
                    numIters=iterations
                )
            except AttributeError:
                raise RuntimeError("OpenCV CUDA acceleration or opencv-contrib (DualTVL1OpticalFlow) is required for TV-L1.")

    def compute_flow_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Computes TV-L1 optical flow between two frames.
        Inputs: frame1, frame2 (H x W x C, uint8 RGB or grayscale).
        Returns: flow (H x W x 2) numpy array.
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1
            
        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2

        if self.use_cuda:
            gpu_frame1 = cv2.cuda_GpuMat()
            gpu_frame1.upload(gray1)
            gpu_frame2 = cv2.cuda_GpuMat()
            gpu_frame2.upload(gray2)
            
            gpu_flow = self.tvl1.calc(gpu_frame1, gpu_frame2, None)
            flow = gpu_flow.download()
        else:
            flow = self.tvl1.calc(gray1, gray2, None)
            
        return flow

    def extract_sequence(self, clip_frames: list) -> list:
        """
        Extracts 5 consecutive TV-L1 optical flow fields for a clip of K=6 frames.
        Input: clip_frames (list of 6 frames).
        Returns: list of 5 flow arrays, each of shape (H, W, 2).
        """
        flows = []
        for i in range(len(clip_frames) - 1):
            flow = self.compute_flow_pair(clip_frames[i], clip_frames[i+1])
            flows.append(flow)
        return flows