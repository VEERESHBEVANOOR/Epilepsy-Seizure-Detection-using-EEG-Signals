# src/converter.py
import cv2
import numpy as np
import pandas as pd
import os

def eeg_image_to_dataframe(image_path: str) -> pd.DataFrame:
    """
    Final, simplified converter. It finds the darkest point in each
    column to trace the signal, which is robust for different image types.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        h, w = image.shape
        waveform = []

        # For each vertical column, find the darkest pixel (the wave)
        for x in range(w):
            column = image[:, x]
            
            if np.min(column) < 255: # Ensure there is some signal
                y_val = np.argmin(column)
                
                # Normalize the value from 0 to 1
                normalized_val = (h - y_val) / h
                waveform.append(normalized_val)
            else:
                waveform.append(np.nan)

        if not waveform:
            return pd.DataFrame()

        df = pd.DataFrame({'Channel_1': waveform})
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        
        print(f"✅ Final Converter Processed: {os.path.basename(image_path)}")
        return df

    except Exception as e:
        print(f"Error in the final converter: {e}")
        return pd.DataFrame()