"""
Post-processing utilities for panel segmentation

Methods to separate touching/merged panels:
1. Morphological opening + Connected Components
2. Watershed transform
"""

import numpy as np
import cv2
from scipy import ndimage


def separate_panels_morphological(mask, kernel_size=5, min_area=500):
    """
    Separate touching panels using morphological opening + connected components
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        kernel_size: Size of erosion/dilation kernel
        min_area: Minimum panel area to keep (removes noise)
    
    Returns:
        separated_mask: Binary mask with separated panels
        labeled_mask: Each panel has unique label (1, 2, 3, ...)
        num_panels: Number of detected panels
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)
    
    # Morphological opening (erosion followed by dilation)
    # This separates touching regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Connected components
    num_labels, labeled_mask = cv2.connectedComponents(opened)
    
    # Filter small components and reconstruct
    separated_mask = np.zeros_like(binary)
    valid_labels = []
    
    for label in range(1, num_labels):  # Skip background (0)
        component = (labeled_mask == label).astype(np.uint8)
        area = component.sum()
        
        if area >= min_area:
            # Dilate back to approximate original size
            dilated = cv2.dilate(component, kernel, iterations=1)
            separated_mask = np.maximum(separated_mask, dilated)
            valid_labels.append(label)
    
    # Re-label with valid labels only
    final_labeled = np.zeros_like(labeled_mask)
    for i, label in enumerate(valid_labels, 1):
        final_labeled[labeled_mask == label] = i
    
    return separated_mask * 255, final_labeled, len(valid_labels)


def separate_panels_watershed(mask, min_distance=20, min_area=500):
    """
    Separate touching panels using distance transform + watershed
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        min_distance: Minimum distance between panel centers for separation
        min_area: Minimum panel area to keep
    
    Returns:
        separated_mask: Binary mask with separated panels
        labeled_mask: Each panel has unique label (1, 2, 3, ...)
        num_panels: Number of detected panels
    """
    # Ensure binary
    binary = (mask > 127).astype(np.uint8)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    # Normalize for visualization
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find local maxima as markers (sure foreground)
    # Use adaptive threshold on distance transform
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Find sure background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labeling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all labels so background is 1 instead of 0
    markers = markers + 1
    
    # Mark unknown region as 0
    markers[unknown == 255] = 0
    
    # Apply watershed
    # Need 3-channel image for watershed
    img_color = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Create separated mask
    separated_mask = np.zeros_like(binary)
    labeled_mask = np.zeros_like(markers)
    
    valid_label = 0
    for label in range(2, num_labels + 1):  # Skip background (1) and boundary (-1)
        component = (markers == label).astype(np.uint8)
        area = component.sum()
        
        if area >= min_area:
            valid_label += 1
            separated_mask[markers == label] = 1
            labeled_mask[markers == label] = valid_label
    
    return separated_mask * 255, labeled_mask, valid_label


def separate_panels_combined(mask, erosion_kernel=5, min_distance=15, min_area=500):
    """
    Combined approach: morphological + watershed for robust separation
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        erosion_kernel: Kernel size for initial erosion
        min_distance: Minimum distance for watershed markers
        min_area: Minimum panel area to keep
    
    Returns:
        separated_mask: Binary mask with separated panels  
        labeled_mask: Each panel has unique label
        num_panels: Number of detected panels
    """
    binary = (mask > 127).astype(np.uint8)
    
    # Step 1: Light erosion to separate barely touching panels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel))
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    # Step 2: Distance transform on eroded mask
    dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    
    # Step 3: Find markers using local maxima
    # Threshold to find sure foreground
    threshold = 0.4 * dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, threshold, 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Step 4: Connected components on sure foreground as markers
    num_markers, markers = cv2.connectedComponents(sure_fg)
    
    # Step 5: Prepare for watershed
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Step 6: Watershed
    img_color = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    # Step 7: Extract and filter panels
    separated_mask = np.zeros_like(binary)
    labeled_mask = np.zeros(markers.shape, dtype=np.int32)
    
    valid_label = 0
    for label in range(2, num_markers + 1):
        component = (markers == label).astype(np.uint8)
        area = component.sum()
        
        if area >= min_area:
            valid_label += 1
            # Dilate slightly to recover eroded area
            recovered = cv2.dilate(component, kernel, iterations=1)
            # But constrain to original mask
            recovered = recovered & binary
            separated_mask = np.maximum(separated_mask, recovered)
            labeled_mask[markers == label] = valid_label
    
    return separated_mask * 255, labeled_mask, valid_label


def visualize_panels(image, labeled_mask, num_panels):
    """
    Create visualization with colored panels
    
    Args:
        image: Original grayscale image (H, W)
        labeled_mask: Labeled mask where each panel has unique ID
        num_panels: Number of panels
    
    Returns:
        visualization: Color image with panels colored differently
    """
    # Generate random colors for each panel
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_panels + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    # Create color visualization
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # Overlay colored panels
    overlay = np.zeros_like(vis)
    for label in range(1, num_panels + 1):
        mask = (labeled_mask == label)
        overlay[mask] = colors[label]
    
    # Blend with original
    result = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    
    # Draw panel boundaries
    for label in range(1, num_panels + 1):
        mask = (labeled_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result


def get_panel_bboxes(labeled_mask, num_panels):
    """
    Extract bounding boxes for each panel
    
    Args:
        labeled_mask: Labeled mask
        num_panels: Number of panels
    
    Returns:
        bboxes: List of (x, y, w, h) for each panel
    """
    bboxes = []
    for label in range(1, num_panels + 1):
        mask = (labeled_mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            bboxes.append((x, y, w, h))
    return bboxes
