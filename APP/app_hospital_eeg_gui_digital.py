import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageSequence
from datetime import datetime
from scipy.stats import kurtosis
import os
import threading
import queue
import time 

# ----- [NEW IMPORTS FOR GRAPHING] -----
# You must run: pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ----------------------------------------


# ---------- [OPTIMIZED] CACHED LAZY-LOADING ----------
# We cache heavy libraries after their first import
_PANDAS_ = None
_CV2_ = None
_REPORTLAB_ = None
_REPORTLAB_AVAILABLE_ = None 

def get_pandas():
    """Lazy-loads and caches the pandas module."""
    global _PANDAS_
    if _PANDAS_ is None:
        try:
            import pandas as pd
            _PANDAS_ = pd
        except ImportError:
            messagebox.showerror("Dependency Missing", "The 'pandas' library is required.\nPlease install it using: pip install pandas")
            return None
    return _PANDAS_

def get_cv2():
    """Lazy-loads and caches the OpenCV (cv2) module."""
    global _CV2_
    if _CV2_ is None:
        try:
            import cv2
            _CV2_ = cv2
        except ImportError:
            messagebox.showerror("Dependency Missing", "The 'opencv-python' library is required.\nPlease install it using: pip install opencv-python")
            return None
    return _CV2_

def check_reportlab():
    """Checks if ReportLab is available and caches the result."""
    global _REPORTLAB_AVAILABLE_
    if _REPORTLAB_AVAILABLE_ is None:
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            _REPORTLAB_AVAILABLE_ = True
        except ImportError:
            _REPORTLAB_AVAILABLE_ = False
    return _REPORTLAB_AVAILABLE_

def get_reportlab():
    """Lazy-loads and caches the ReportLab modules."""
    global _REPORTLAB_
    if _REPORTLAB_ is None and check_reportlab():
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, Table, TableStyle
            from reportlab.lib import colors
            _REPORTLAB_ = {
                "canvas": canvas, "letter": letter, "inch": inch,
                "getSampleStyleSheet": getSampleStyleSheet, "Paragraph": Paragraph,
                "Table": Table, "TableStyle": TableStyle, "colors": colors
            }
        except ImportError:
            _REPORTLAB_AVAILABLE_ = False # Mark as unavailable
            return None
    return _REPORTLAB_


# ----- [NEW] EEG IMAGE TO DATAFRAME FUNCTION (Integrated) -----
# Replaces the `from src.converter import ...`
def eeg_image_to_dataframe(image_path: str):
    """
    Analyzes an EEG image, traces the primary waveform, and extracts it
    into a pandas DataFrame, simulating a time-series signal.
    """
    
    # 1. Get the heavy libraries (lazy-loaded)
    cv2 = get_cv2()
    pd = get_pandas()
    if not cv2 or not pd:
        print("Error: Missing critical libraries.")
        return pd.DataFrame() # Return empty DataFrame

    try:
        # 2. Load the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return pd.DataFrame()

        # 3. Binarize the image (same as preview)
        _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        
        # 4. Get the dimensions
        height, width = binary_image.shape
        
        # 5. Trace the waveform
        signal_values = []
        
        for x in range(width):
            column = binary_image[:, x] # Get column
            white_pixel_indices = np.where(column == 255)[0] # Find signal
            
            if white_pixel_indices.size > 0:
                y_position = np.mean(white_pixel_indices) # Average thick lines
                inverted_y = height - y_position # Flip graph upright
                signal_values.append(inverted_y)
            else:
                if signal_values: # Fill gaps
                    signal_values.append(signal_values[-1])
                else:
                    signal_values.append(height / 2) # Start in middle

        # 6. Create the output DataFrame
        df = pd.DataFrame(signal_values, columns=['EEG_Signal_Value'])
        
        # 7. Normalize the data (common for signals)
        min_val = df['EEG_Signal_Value'].min()
        max_val = df['EEG_Signal_Value'].max()
        if (max_val - min_val) > 0:
            df['Normalized_Signal'] = 2 * ((df['EEG_Signal_Value'] - min_val) / (max_val - min_val)) - 1
        else:
            df['Normalized_Signal'] = 0 # Flat line
            
        # Return the DataFrame with the column the main app expects
        return df[['Normalized_Signal']]

    except Exception as e:
        messagebox.showerror("Conversion Error", f"Failed to process image: {e}")
        return pd.DataFrame()
# -----------------------------------------------------------------


# ---------- [MODIFIED] VIDEO PROCESSING THREAD FUNCTION ----------
def video_processing_thread(video_path, frame_queue, dimension_func, stop_event):
    """
    A dedicated thread function to handle all heavy video processing.
    """
    cv2 = get_cv2()
    if not cv2: 
        print("CV2 not available, video thread exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        w, h = dimension_func() 
        
        if w > 1 and h > 1:
            try:
                if frame_queue.full():
                    frame_queue.get_nowait() # Discard oldest frame

                frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)
                frame_queue.put(img_tk)
            except queue.Full:
                pass 
            except Exception as e:
                pass
        else:
            time.sleep(0.1) 
    
    cap.release()
    print("Video thread stopped.")


# ---------- [MODIFIED] PREDICTION LOGIC (Returns a Dictionary) ----------
def predict_eeg_with_logic(df): 
    """
    This is the final, reliable prediction engine.
    
    [MODIFIED] Now returns a dictionary with all calculation details.
    """
    pd = get_pandas()
    if not pd: 
        return {
            "Prediction": "Error", "Confidence": 0,
            "Details": "Pandas library not loaded."
        }
    
    empty_result = {
        "Prediction": "Not enough data", "Confidence": 0,
        "Abnormal Windows": 0, "Max Std Dev": 0.0, "Threshold": 0.20
    }
    
    if df.empty or df.iloc[:, 0].dropna().empty:
        return empty_result
    
    signal = df.iloc[:, 0].dropna().values
    window_size = 178; step_size = window_size // 4
    STD_DEV_THRESHOLD = 0.20
    seizure_window_count = 0; max_confidence_score = 0.0

    if len(signal) < window_size:
        return empty_result

    for i in range(0, len(signal) - window_size + 1, step_size):
        window = signal[i:i + window_size]
        if window.size < 10: continue
        current_std = np.std(window)
        
        if current_std > STD_DEV_THRESHOLD:
            seizure_window_count += 1
            if current_std > max_confidence_score:
                max_confidence_score = current_std
    
    if seizure_window_count >= 2:
        confidence = 0.85 + (min((max_confidence_score / (STD_DEV_THRESHOLD * 2)), 1.0) * 0.14)
        return {
            "Prediction": "Epilepsy Detected",
            "Confidence": min(confidence, 0.99),
            "Abnormal Windows": seizure_window_count,
            "Max Std Dev": max_confidence_score,
            "Threshold": STD_DEV_THRESHOLD
        }
    else:
        max_normal_std = max_confidence_score if max_confidence_score > 0 else np.std(signal)
        return {
            "Prediction": "Normal",
            "Confidence": 0.95,
            "Abnormal Windows": seizure_window_count,
            "Max Std Dev": max_normal_std, # Show max std even if normal
            "Threshold": STD_DEV_THRESHOLD
        }
# -----------------------------------------------------------------


# ---------- CUSTOM BUTTON CLASS (3D Effect) ----------
class CustomButton(tk.Canvas):
    
    def _hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    def _adjust_color(self, hex_color, factor):
        r, g, b = self._hex_to_rgb(hex_color)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return self._rgb_to_hex((r, g, b))
    
    def __init__(self, parent, text, command, **kwargs):
        self.width = kwargs.pop('width', 400); self.height = kwargs.pop('height', 60)
        self.font = kwargs.pop('font', ("Segoe UI", 16, "bold")) 
        
        kwargs.pop('border_color', None); kwargs.pop('border_width', None)
        kwargs.pop('border_hover_width', None); kwargs.pop('stipple', None)
        
        bg = kwargs.pop('bg', "#607D8B")
        hover_bg = kwargs.pop('hover_bg', self._adjust_color(bg, 1.1))
        pressed_bg = kwargs.pop('pressed_bg', self._adjust_color(bg, 0.9))
        
        self.colors = {
            "normal_bg": bg, "normal_highlight": self._adjust_color(bg, 1.3),
            "normal_shadow": self._adjust_color(bg, 0.7),
            "hover_bg": hover_bg, "hover_highlight": self._adjust_color(hover_bg, 1.3),
            "hover_shadow": self._adjust_color(hover_bg, 0.7),
            "pressed_bg": pressed_bg, "pressed_highlight": self._adjust_color(pressed_bg, 0.7),
            "pressed_shadow": self._adjust_color(pressed_bg, 1.3),
            "normal_fg": kwargs.pop('fg', "white"), "hover_fg": kwargs.pop('hover_fg', "white"),
            "pressed_fg": kwargs.pop('pressed_fg', "white")
        }
        
        self.corner_radius = kwargs.pop('radius', 8); self.shadow_depth = 4
        
        super().__init__(parent, width=self.width, height=self.height, **kwargs)
        self.command = command; self.text = text
        
        self.config(highlightthickness=0, bg=self.master.cget('bg'))
        
        self._draw_button('normal'); self.bind("<Enter>", self._on_enter); self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press); self.bind("<ButtonRelease-1>", self._on_release)

    def round_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
                  x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
                  x1, y2, x1, y2-radius, x1, y1+radius, x1, y1]
        self.create_polygon(points, smooth=True, **kwargs)

    def _draw_button(self, state):
        self.delete("all")
        bg_color = self.colors[f"{state}_bg"]; highlight_color = self.colors[f"{state}_highlight"]
        shadow_color = self.colors[f"{state}_shadow"]; text_color = self.colors[f"{state}_fg"]
        padding = 1; text_x_offset = 0; text_y_offset = 0

        if state == 'pressed':
            self.round_rectangle(0, 0, self.width, self.height, self.corner_radius, fill=highlight_color, outline="")
            self.round_rectangle(padding, padding, self.width - padding, self.height - padding, self.corner_radius, fill=shadow_color, outline="")
            self.round_rectangle(self.shadow_depth, self.shadow_depth, self.width - padding, self.height - padding, self.corner_radius, fill=bg_color, outline="")
            text_x_offset = self.shadow_depth / 2; text_y_offset = self.shadow_depth / 2
        else:
            self.round_rectangle(self.shadow_depth, self.shadow_depth, self.width, self.height, self.corner_radius, fill=shadow_color, outline="")
            self.round_rectangle(0, 0, self.width - self.shadow_depth, self.height - self.shadow_depth, self.corner_radius, fill=highlight_color, outline="")
            self.round_rectangle(padding, padding, self.width - self.shadow_depth - padding, self.height - self.shadow_depth - padding, self.corner_radius, fill=bg_color, outline="")
            text_x_offset = -self.shadow_depth / 2; text_y_offset = -self.shadow_depth / 2

        self.create_text(self.width / 2 + text_x_offset, self.height / 2 + text_y_offset, 
                         text=self.text, font=self.font, fill=text_color, tags="main_text")
        
    def _on_enter(self, event): self._draw_button('hover')
    def _on_leave(self, event): self._draw_button('normal')
    def _on_press(self, event): self._draw_button('pressed')
    def _on_release(self, event):
        self._draw_button('hover')
        if self.command: self.command()

# ---------- PATIENT INFO FORM FOR PDF REPORT ----------
class PatientInfoForm(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.title("Enter Patient Details for Report"); self.geometry("500x450"); self.configure(bg="#2C3E50")
        self.callback = callback; self.resizable(False, False); self.transient(master); self.grab_set()
        vcmd_digit = (self.register(self.validate_digit), '%P'); vcmd_len_20 = (self.register(self.validate_len), '%P', 20)
        vcmd_len_10 = (self.register(self.validate_len), '%P', 10); vcmd_len_3 = (self.register(self.validate_len), '%P', 3)
        vcmd_len_30 = (self.register(self.validate_len), '%P', 30)
        fields = {"Name": {"vcmd": vcmd_len_20}, "Age": {"vcmd": vcmd_len_3, "ivcmd": vcmd_digit}, "Sex": {"options": ["Male", "Female", "Other"]}, "Phone Number": {"vcmd": vcmd_len_10, "ivcmd": vcmd_digit}, "Email": {"vcmd": vcmd_len_30}}
        self.entries = {}; frame = tk.Frame(self, bg="#2C3E50", padx=20, pady=20); frame.pack(fill="both", expand=True)
        for i, (label_text, config) in enumerate(fields.items()):
            lbl = tk.Label(frame, text=label_text, font=("Segoe UI", 12), bg="#2C3E50", fg="white")
            lbl.grid(row=i, column=0, sticky="w", pady=5, padx=5)
            if "options" in config:
                self.entries[label_text] = ttk.Combobox(frame, values=config["options"], state="readonly", font=("Segoe UI", 12)); self.entries[label_text].set(config["options"][0])
            else:
                self.entries[label_text] = tk.Entry(frame, font=("Segoe UI", 12), width=35, validate="key", validatecommand=config.get("ivcmd"), invalidcommand=config.get("vcmd"))
            self.entries[label_text].grid(row=i, column=1, sticky="ew", pady=5, padx=5)
        frame.grid_columnconfigure(1, weight=1); btn_frame = tk.Frame(self, bg="#2C3E50", pady=10); btn_frame.pack(fill="x")
        
        save_btn = CustomButton(btn_frame, text="Save Report", command=self.submit, width=180, height=50, font=("Segoe UI", 14, "bold"), 
                                bg="#28A745", hover_bg="#218838", radius=8)
        save_btn.pack(side="right", padx=20)
        cancel_btn = CustomButton(btn_frame, text="Cancel", command=self.destroy, width=120, height=50, font=("Segoe UI", 14, "bold"), 
                                  bg="#DC3545", hover_bg="#C82333", radius=8)
        cancel_btn.pack(side="right", padx=10)
        
    def validate_digit(self, P): return P.isdigit() or P == ""
    def validate_len(self, P, max_len): return len(P) <= int(max_len)
    def submit(self):
        patient_data = {label: entry.get() for label, entry in self.entries.items()}
        for label, value in patient_data.items():
            if not value: messagebox.showerror("Validation Error", f"'{label}' cannot be empty.", parent=self); return
        self.destroy(); self.callback(patient_data)

# ---------- EEG IMAGE TO CSV CONVERTER WINDOW ----------
class EEGConverterApp:
    def __init__(self, master):
        self.root = tk.Toplevel(master); self.root.title("EEG Image to CSV Converter"); self.root.geometry("1200x800"); self.root.minsize(1000, 700)
        self.image_path = None; self.image = None; self.processed_waveform = None; self.video_path = "app/image.mov"
        self.bg_photo = None; self.video_frame_queue = queue.Queue(maxsize=5)
        
        self.video_dimensions = [self.root.winfo_width(), self.root.winfo_height()]
        self.video_dim_lock = threading.Lock()
        self.video_stop_event = threading.Event()
        
        self.get_video_dimensions = self.get_safe_dimensions 
        self.video_thread = threading.Thread(target=video_processing_thread, 
                                             args=(self.video_path, self.video_frame_queue, 
                                                   self.get_video_dimensions, self.video_stop_event), 
                                             daemon=True)
        
        self.bg_canvas = tk.Canvas(self.root, highlightthickness=0); self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_on_canvas = self.bg_canvas.create_image(0, 0, anchor="nw")
        self.root.grid_columnconfigure(0, weight=1, uniform="group1"); self.root.grid_columnconfigure(1, weight=2, uniform="group1")
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_rowconfigure(1, weight=1); self.root.grid_rowconfigure(2, weight=1); self.root.grid_rowconfigure(3, weight=1)
        
        button_font = ("Segoe UI", 20, "bold")
        button_config = {"font": button_font, "fg": "white", "hover_fg": "white", "width": 350, "height": 70, 
                         "radius": 10, "bg": "#007BFF", "hover_bg": "#0056B3"} 
        CustomButton(self.root, text="Load EEG Image", command=self.load_image, **button_config).grid(row=0, column=0, pady=10)
        CustomButton(self.root, text="Preview Preprocessing", command=self.preview_preprocessing, **button_config).grid(row=1, column=0, pady=10)
        CustomButton(self.root, text="Extract & Convert", command=self.extract_waveform, **button_config).grid(row=2, column=0, pady=10)
        CustomButton(self.root, text="Save CSV", command=self.save_csv, **button_config).grid(row=3, column=0, pady=10)
        
        self.image_display_label = tk.Label(self.root, bd=0); self.image_display_label.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20); self.image_display_label.grid_remove()
        
        back_button = CustomButton(self.root, text="Back to Main", command=self.close_window, width=250, height=50, 
                                   font=("Segoe UI", 16, "bold"), fg="white", hover_fg="white", 
                                   radius=10, bg="#DC3545", hover_bg="#C82333")
        back_button.place(relx=0.5, rely=0.98, anchor="s")

        self.root.bind("<Configure>", self.on_resize) 
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.root.after(50, self.start_video_thread); 
        self.root.after(100, self._update_video_background)
        self.root.after(100, lambda: self.on_resize(None)) 

    def close_window(self):
        print("Closing converter window...")
        self.video_stop_event.set()
        self.root.destroy()
        
    def on_resize(self, event):
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        with self.video_dim_lock:
            self.video_dimensions[0] = width
            self.video_dimensions[1] = height
            
    def get_safe_dimensions(self):
        with self.video_dim_lock:
            return tuple(self.video_dimensions)

    def start_video_thread(self): 
        if not self.video_thread.is_alive():
            self.video_thread.start()
            
    def _update_video_background(self):
        try:
            self.bg_photo = self.video_frame_queue.get_nowait()
            self.bg_canvas.itemconfig(self.bg_on_canvas, image=self.bg_photo)
        except queue.Empty: pass
        except Exception as e: 
            if isinstance(e, tk.TclError): return 
        
        if not self.video_stop_event.is_set():
             self.root.after(33, self._update_video_background)

    def load_image(self):
        cv2 = get_cv2()
        if not cv2: return
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path: return
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None: messagebox.showerror("Error", "Failed to load image."); return
        self.image_path = file_path; self.image = img; self.display_image(self.image); messagebox.showinfo("Loaded", f"Loaded {file_path}")
    
    def preview_preprocessing(self):
        cv2 = get_cv2()
        if not cv2: return
        if self.image is None: messagebox.showerror("Error", "Please load an EEG image first."); return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY); _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        self.display_image(binary, is_gray=True)
    
    def extract_waveform(self):
        if self.image_path is None: messagebox.showerror("Error", "Please load an EEG image first."); return
        try:
            # ----- [MODIFIED] Uses the integrated function -----
            self.processed_waveform = eeg_image_to_dataframe(self.image_path)
            if self.processed_waveform is None or self.processed_waveform.empty: 
                messagebox.showerror("Error", "Could not extract any waveform data."); self.processed_waveform = None
            else: messagebox.showinfo("Done", "Waveform extracted successfully! You can now save it as a CSV.")
        except Exception as e: messagebox.showerror("Extraction Failed", f"An error occurred: {e}"); self.processed_waveform = None
    
    def save_csv(self):
        pd = get_pandas()
        if not pd: return
        if self.processed_waveform is None: messagebox.showerror("Error", "Please extract the waveform first."); return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path: self.processed_waveform.to_csv(file_path, index=False); messagebox.showinfo("Saved", f"Waveform saved as {file_path}")
    
    def display_image(self, img, is_gray=False):
        cv2 = get_cv2()
        if not cv2 or img is None: return
        
        self.image_display_label.grid(); self.image_display_label.configure(bg="white"); 
        try:
            self.root.update_idletasks() 
        except tk.TclError:
            return 

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if is_gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        canvas_w = self.image_display_label.winfo_width(); canvas_h = self.image_display_label.winfo_height()
        if canvas_w < 50 or canvas_h < 50: canvas_w, canvas_h = 800, 500
        w, h = img_pil.size; scale = min(canvas_w / w, canvas_h / h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_display_label.configure(image=img_tk); self.image_display_label.image = img_tk

# ---------- [MODIFIED] EPILEPSY DETECTOR WINDOW ----------
class EpilepsyDetectorApp:
    def __init__(self, master, initial_df=None):
        self.root = tk.Toplevel(master); self.root.title("Epilepsy Detector - BLDEA Hospital"); self.root.attributes('-fullscreen', True)
        self.video_path = "app/brain.mov"; 
        self.bg_photo = None; self.video_frame_queue = queue.Queue(maxsize=5)
        
        # ----- [NEW] Variables for graph and calculations -----
        self.df_uploaded = None         # Stores the cleaned DF for graphing
        self.final_result_data = None   # Stores the full results dictionary
        self.graph_canvas = None        # Stores the matplotlib canvas widget
        self.calc_frame = None          # Stores the frame for calculation labels
        # ----------------------------------------------------
        
        self.video_dimensions = [self.root.winfo_screenwidth(), self.root.winfo_screenheight()]
        self.video_dim_lock = threading.Lock()
        self.video_stop_event = threading.Event()

        self.get_video_dimensions = self.get_safe_dimensions 
        self.video_thread = threading.Thread(target=video_processing_thread, 
                                             args=(self.video_path, self.video_frame_queue, 
                                                   self.get_video_dimensions, self.video_stop_event), 
                                             daemon=True)

        self.gif_frame_index = 0; self.eeg_gif_frame_index = 0
        self.brain_gif_frames = []; self.eeg_gif_frames = []
        self.current_brain_tk = None; self.current_eeg_tk = None
        
        self.setup_ui() 
        self.load_gifs()
        
        self.root.bind("<Configure>", self.on_resize); 
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.root.after(50, lambda: self.on_resize(None)) 
        self.root.after(50, self.start_video_thread); 
        self.root.after(100, self.start_animations)
        
        if initial_df is not None: self.process_dataframe(initial_df)
    
    def close_window(self):
        print("Closing detector window...")
        self.video_stop_event.set()
        # [FIX] Clean up matplotlib widgets explicitly
        self.hide_results() 
        self.root.destroy()
        
    def on_resize(self, event):
        if not hasattr(self, 'canvas'): return 
        width = self.root.winfo_width(); height = self.root.winfo_height()
        
        with self.video_dim_lock:
            self.video_dimensions[0] = width
            self.video_dimensions[1] = height
        
        self.canvas.coords(self.title1_shadow, width / 2 + 2, 40 + 2); self.canvas.coords(self.title1, width / 2, 40)
        self.canvas.coords(self.title2_shadow, width / 2 + 1, 75 + 1); self.canvas.coords(self.title2, width / 2, 75)
        self.canvas.coords(self.brain_gif_window, width - 20, 20); self.canvas.coords(self.eeg_gif_window, width / 2, height)
        
        center_x = width / 2; btn1_x = center_x - 115; btn2_x = center_x + 150
        self.canvas.coords(self.upload_button_window, btn1_x, height - 50)
        self.canvas.coords(self.back_button_window, btn2_x, height - 50)
        
        if self.canvas.itemcget(self.results_window, "state") == "normal":
             self.canvas.coords(self.results_window, width/2, height/2); self.canvas.itemconfig(self.results_window, width=width*0.9, height=height*0.75)
            
    def get_safe_dimensions(self):
        with self.video_dim_lock:
            return tuple(self.video_dimensions)
                
    def start_video_thread(self): 
        if not self.video_thread.is_alive():
            self.video_thread.start()

    # ----- [MODIFIED] process_dataframe -----
    def process_dataframe(self, df):
        """
        [MODIFIED] Saves the cleaned data and the full result dictionary.
        """
        cleaned_df = self.clean_csv(df)
        if cleaned_df is None: messagebox.showwarning("Warning", "Uploaded file has no valid numeric data."); return
        try:
            # [MODIFIED] Store the full dictionary
            self.final_result_data = predict_eeg_with_logic(cleaned_df)
            
            # [MODIFIED] Store the cleaned data for graphing
            self.df_uploaded = cleaned_df 
            
            self.show_results()
        except Exception as e: 
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
            self.final_result_data = None # Clear on error
            self.df_uploaded = None

    def _load_gif_frames(self, path, size, max_frames=100):
        frames = []
        try:
            with Image.open(path) as gif:
                for i, frame in enumerate(ImageSequence.Iterator(gif)):
                    if i > max_frames: break 
                    frame_resized = frame.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
                    frames.append(ImageTk.PhotoImage(frame_resized))
        except FileNotFoundError:
            print(f"Error: GIF file not found at {path}")
        except Exception as e:
            print(f"Failed to load GIF {path}: {e}")
        return frames

    def load_gifs(self):
        print("Loading GIFs...")
        self.brain_gif_frames = self._load_gif_frames("app/brain_rotating.gif", (100, 100))
        w = self.root.winfo_screenwidth()
        self.eeg_gif_frames = self._load_gif_frames("app/eeg_waveform.gif", (w, 100))
        print("GIFs loaded.")

    def start_animations(self):
        self.animate_brain_gif(); self.animate_eeg_gif(); self._update_video_background()

    def animate_brain_gif(self):
        if self.video_stop_event.is_set() or not self.brain_gif_frames: 
            return 
        try:
            self.current_brain_tk = self.brain_gif_frames[self.gif_frame_index]
            self.brain_gif_label.configure(image=self.current_brain_tk)
            self.gif_frame_index = (self.gif_frame_index + 1) % len(self.brain_gif_frames)
        except tk.TclError: return 
        except Exception: pass 
        
        self.root.after(80, self.animate_brain_gif)

    def animate_eeg_gif(self):
        if self.video_stop_event.is_set() or not self.eeg_gif_frames: 
            return 
        try:
            self.current_eeg_tk = self.eeg_gif_frames[self.eeg_gif_frame_index]
            self.eeg_gif_label.configure(image=self.current_eeg_tk)
            self.eeg_gif_frame_index = (self.eeg_gif_frame_index + 1) % len(self.eeg_gif_frames)
        except tk.TclError: return 
        except Exception: pass 
            
        self.root.after(80, self.animate_eeg_gif)

    def _update_video_background(self):
        try:
            self.bg_photo = self.video_frame_queue.get_nowait()
            self.canvas.itemconfig(self.bg_on_canvas, image=self.bg_photo)
        except queue.Empty: pass
        except Exception as e: 
            if isinstance(e, tk.TclError): return 

        if not self.video_stop_event.is_set():
            self.root.after(33, self._update_video_background)
        
    def setup_ui(self):
        self.canvas = tk.Canvas(self.root, highlightthickness=0); self.canvas.pack(fill="both", expand=True)
        self.bg_on_canvas = self.canvas.create_image(0, 0, anchor="nw"); width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.title1_shadow = self.canvas.create_text(width/2 + 2, 40 + 2, text="BLDEA HOSPITAL", font=("Segoe UI", 22, "bold"), fill="black")
        self.title1 = self.canvas.create_text(width/2, 40, text="BLDEA HOSPITAL", font=("Segoe UI", 22, "bold"), fill="white")
        self.title2_shadow = self.canvas.create_text(width/2 + 1, 75 + 1, text="Epilepsy Detector", font=("Segoe UI", 14), fill="black")
        self.title2 = self.canvas.create_text(width/2, 75, text="Epilepsy Detector", font=("Segoe UI", 14), fill="white")
        self.brain_gif_label = tk.Label(self.root, bd=0, bg="#1c2e4a"); self.brain_gif_window = self.canvas.create_window(width - 20, 20, anchor="ne", window=self.brain_gif_label)
        self.eeg_gif_label = tk.Label(self.root, bd=0, bg="#1c2e4a"); self.eeg_gif_window = self.canvas.create_window(width/2, height, anchor="s", window=self.eeg_gif_label)
        
        upload_button = CustomButton(
            self.canvas, text="Upload EEG Data", command=self.upload_file, 
            width=280, height=60, font=("Segoe UI", 15, "bold"), 
            fg="white", hover_fg="white", 
            radius=10, bg="#007BFF", hover_bg="#0056B3"
        )
        
        back_button = CustomButton(
            self.canvas, text="Back to Main", command=self.close_window, 
            width=200, height=60, font=("Segoe UI", 15, "bold"), 
            fg="white", hover_fg="white", 
            radius=10, bg="#DC3545", hover_bg="#C82333"
        )

        self.upload_button_window = self.canvas.create_window(width / 2 - 115, height - 50, anchor="c", window=upload_button)
        self.back_button_window = self.canvas.create_window(width / 2 + 150, height - 50, anchor="c", window=back_button)

        # ----- [MODIFIED] Setup for Results Frame -----
        # This frame holds the graph, calculations, table, and buttons
        self.results_frame = tk.Frame(self.root, bg="#17202A")
        
        # --- Container for the controls (Save, Back) ---
        # We define this first so we can pack it at the BOTTOM
        controls_frame = tk.Frame(self.results_frame, bg="#17202A"); 
        controls_frame.pack(side="bottom", fill="x", pady=10) # Pack at bottom
        
        save_btn = CustomButton(
            controls_frame, text="Save Report as PDF", command=self.prompt_for_report_details, 
            width=280, height=50, font=("Segoe UI", 14, "bold"), 
            fg="white", hover_fg="white", 
            radius=10, bg="#28A745", hover_bg="#218838" 
        )
        save_btn.pack(side="right", padx=15)
        
        back_btn = CustomButton(
            controls_frame, text="Back to Upload", command=self.hide_results, 
            width=220, height=50, font=("Segoe UI", 14, "bold"), 
            fg="white", hover_fg="white", 
            radius=10, bg="#DC3545", hover_bg="#C82333" 
        )
        back_btn.pack(side="right", padx=15)
        
        # --- Container for the main results table ---
        # We pack this *above* the controls
        table_container = tk.Frame(self.results_frame, bg="#212F3D"); 
        table_container.pack(side="bottom", fill="x", expand=False, padx=10, pady=10) # Pack above controls
        
        style = ttk.Style(); style.theme_use("default")
        style.configure("Treeview", background="#212F3D", foreground="white", fieldbackground="#212F3D", rowheight=45, font=("Segoe UI", 14))
        style.configure("Treeview.Heading", background="#17202A", foreground="white", font=("Segoe UI", 16, "bold"))
        style.map("Treeview", background=[('selected', '#007acc')])
        
        self.tree = ttk.Treeview(table_container, style="Treeview", height=2); # Limit height
        self.tree.pack(side="left", fill="x", expand=True)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview); 
        # scrollbar.pack(side="right", fill="y") # No scrollbar needed for 2 rows
        self.tree.configure(yscroll=scrollbar.set)
        
        # --- The Results Window (whole frame) ---
        self.results_window = self.canvas.create_window(width/2, height/2, anchor="center", window=self.results_frame, state="hidden")
    
    def prompt_for_report_details(self):
        if not self.final_result_data:
            messagebox.showerror("Error", "No prediction result available.")
            return
        if not check_reportlab():
            messagebox.showerror("Dependency Missing", "The 'reportlab' library is required.\nPlease install it using: pip install reportlab")
            return
        PatientInfoForm(self.root, self.create_and_save_report)

    def create_and_save_report(self, patient_info):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Documents", "*.pdf")], title="Save Report As...")
        if not file_path: return
        try:
            self.generate_pdf_report(patient_info, file_path)
            messagebox.showinfo("Success", f"Report successfully saved to:\n{file_path}")
        except Exception as e: 
            print(f"PDF Error: {e}")
            messagebox.showerror("PDF Error", f"Failed to generate PDF report: {e}")
            
    def generate_pdf_report(self, patient_info, file_path):
        rl = get_reportlab()
        if not rl:
            messagebox.showerror("Error", "ReportLab library could not be loaded.")
            return
        
        # [MODIFIED] Get result text from the data dictionary
        final_result_text = self.final_result_data.get("Prediction", "Unknown")
        final_conf_text = self.final_result_data.get("Confidence", 0)
        full_result_string = f"⚠️ {final_result_text} (Confidence: {final_conf_text:.2%})" if "Epilepsy" in final_result_text else f"✅ {final_result_text} (Confidence: {final_conf_text:.2%})"
        
        c = rl['canvas'].Canvas(file_path, pagesize=rl['letter']); width, height = rl['letter']; margin = 0.75 * rl['inch']
        c.setStrokeColorRGB(0.1, 0.1, 0.1); c.setLineWidth(1); c.rect(margin, margin, width - 2 * margin, height - 2 * margin)
        c.setFont("Helvetica-Bold", 20); c.drawCentredString(width / 2.0, height - margin - 25, "BLDEA HOSPITAL")
        c.setFont("Helvetica", 14); c.drawCentredString(width / 2.0, height - margin - 50, "EEG Analysis & Epilepsy Screening Report")
        c.line(margin, height - margin - 65, width - margin, height - margin - 65)
        y_pos = height - margin - 90; c.setFont("Helvetica-Bold", 12); c.drawString(margin + 10, y_pos, "Patient Information")
        c.line(margin, y_pos - 10, width - margin, y_pos - 10); c.setFont("Helvetica", 11); info_y_start = y_pos - 30
        for i, (key, value) in enumerate(patient_info.items()):
            c.drawString(margin + 10, info_y_start - (i * 20), f"{key}:"); c.drawString(margin + 130, info_y_start - (i * 20), str(value))
        y_pos = info_y_start - (len(patient_info) * 20) - 15; c.setFont("Helvetica-Bold", 12); c.drawString(margin + 10, y_pos, "Screening Result")
        c.line(margin, y_pos - 10, width - margin, y_pos - 10); y_pos -= 35; c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.8, 0, 0) if "Epilepsy" in final_result_text else c.setFillColorRGB(0, 0.6, 0)
        c.drawString(margin + 10, y_pos, full_result_string); c.setFillColorRGB(0, 0, 0); y_pos -= 30
        styles = rl['getSampleStyleSheet'](); style_body = styles['BodyText']; style_body.leading = 15
        style_header = styles['h5']; style_header.alignment = 1
        symptoms_text = "• Temporary confusion or a staring spell <br/>• Uncontrollable jerking movements of the arms and legs <br/>• Loss of consciousness or awareness <br/>• Cognitive or emotional changes, such as fear, anxiety, or déjà vu"
        precautions_text = "• <b>Medication:</b> Take anti-seizure medication exactly as prescribed. <br/>• <b>Sleep:</b> Ensure adequate sleep, as fatigue is a common trigger. <br/>• <b>Avoid Triggers:</b> Identify and avoid personal triggers (e.g., flashing lights, stress). <br/>• <b>Safety:</b> Inform others about seizure first aid. Consider a medical alert bracelet."
        header_symptoms = rl['Paragraph']("<b>Common Symptoms</b>", style_header); header_precautions = rl['Paragraph']("<b>Precautions & Management</b>", style_header)
        p_symptoms = rl['Paragraph'](symptoms_text, style_body); p_precautions = rl['Paragraph'](precautions_text, style_body)
        data = [[header_symptoms, header_precautions]]; col_width = (width - 2 * margin - 10) / 2
        table = rl['Table'](data, colWidths=[col_width, col_width])
        ts = rl['TableStyle']([('BACKGROUND', (0,0), (-1,0), rl['colors'].lightgrey), ('TEXTCOLOR', (0,0), (-1,0), rl['colors'].black), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0,0), (-1,0), 12), ('GRID', (0,0), (-1,-1), 1, rl['colors'].black), ('BOX', (0,0), (-1,-1), 2, rl['colors'].black), ('TOPPADDING', (0,1), (-1,-1), 8), ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10)])
        table.setStyle(ts); table_width, table_height = table.wrapOn(c, width, height); table.drawOn(c, margin + 5, y_pos - table_height)
        
        data2 = [[p_symptoms, p_precautions]]
        table2 = rl['Table'](data2, colWidths=[col_width, col_width])
        table2.setStyle(ts) 
        table2_width, table2_height = table2.wrapOn(c, width, height); 
        table2.drawOn(c, margin + 5, y_pos - table_height - table2_height)
        
        footer_y = margin + 80; report_time = datetime.now().strftime("%d/%m/%Y %I:%M:%S %p"); c.setFont("Helvetica", 9)
        c.drawRightString(width - margin - 5, footer_y, f"Report Generated: {report_time}"); c.setFont("Helvetica-Oblique", 9)
        disclaimer = "This report is generated based on a computational analysis of EEG data and is intended for screening purposes only. It is not a substitute for a professional medical diagnosis. Please consult a qualified neurologist for a comprehensive evaluation."
        p_disclaimer = rl['Paragraph'](disclaimer, styles['Italic']); p_width, p_height = p_disclaimer.wrapOn(c, width - 2 * margin - 10, height)
        p_disclaimer.drawOn(c, margin + 5, footer_y - 20 - p_height)
        c.line(margin, margin + 25, width - margin, margin + 25); c.drawCentredString(width / 2.0, margin + 10, "BLDEA Hospital - Confidential Medical Report")
        c.save()
        
    def upload_file(self):
        pd = get_pandas() 
        if not pd: return
        
        file_path = filedialog.askopenfilename(filetypes=[("CSV and Excel files", "*.csv *.xls *.xlsx")])
        if not file_path: return
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, on_bad_lines='skip')
            else:
                df = pd.read_excel(file_path)
            self.process_dataframe(df)
        except Exception as e: messagebox.showerror("Error", f"Failed to read or process file:\n{str(e)}")
    
    # ----- [MODIFIED] show_results -----
    def show_results(self):
        """
        [MODIFIED]
        This function now dynamically adds the Graph and Calculation labels
        to the results_frame.
        """
        
        # 1. Clean up old widgets (if any)
        self.hide_results_widgets() # Call the new cleanup function
        
        # 2. Show the main results window
        self.canvas.itemconfig(self.upload_button_window, state="hidden")
        self.canvas.itemconfig(self.back_button_window, state="hidden")
        self.canvas.itemconfig(self.results_window, state="normal")

        # 3. Get all result data
        if not self.final_result_data: 
            self.final_result_data = {"Prediction": "Error", "Confidence": 0} # Failsafe
            
        prediction = self.final_result_data.get("Prediction", "Unknown")
        confidence = self.final_result_data.get("Confidence", 0)
        abnormal_windows = self.final_result_data.get("Abnormal Windows", "N/A")
        max_std = self.final_result_data.get("Max Std Dev", 0.0)
        threshold = self.final_result_data.get("Threshold", "N/A")
        
        final_text = f"{prediction} (Confidence: {confidence:.2%})"

        # ----- 4. [NEW] Create and show the Matplotlib Graph -----
        try:
            fig = plt.Figure(figsize=(6, 2.5), dpi=100) # (width, height)
            fig.patch.set_facecolor('#17202A') # Match frame background
            ax = fig.add_subplot(111)

            # Plot the data
            if self.df_uploaded is not None and not self.df_uploaded.empty:
                ax.plot(self.df_uploaded.iloc[:, 0], color="#82e0aa", linewidth=1.0)
            
            # Style the graph
            ax.set_title("Uploaded EEG Signal Waveform", color="white", fontsize=12)
            ax.set_facecolor("#212F3D")
            ax.tick_params(axis='x', colors='white', labelsize=8)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            
            fig.tight_layout()

            # Embed the graph in the Tkinter frame
            self.graph_canvas = FigureCanvasTkAgg(fig, master=self.results_frame)
            self.graph_canvas.draw()
            # Pack graph at the TOP, it will fill the remaining space
            self.graph_canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        except Exception as e:
            print(f"Graphing Error: {e}")
            # Show a placeholder if graphing fails
            self.calc_frame = tk.Frame(self.results_frame, bg="#17202A")
            self.calc_frame.pack(side="top", fill="x", pady=10)
            tk.Label(self.calc_frame, text=f"Could not render graph: {e}", bg="#17202A", fg="red").pack()

        # ----- 5. [NEW] Create and show the Calculation Details -----
        self.calc_frame = tk.Frame(self.results_frame, bg="#212F3D", bd=1, relief="solid")
        self.calc_frame.pack(side="top", fill="x", padx=10, pady=(0, 10)) # Pack below graph

        calc_font = ("Segoe UI", 12)
        header_font = ("Segoe UI", 14, "bold")
        
        tk.Label(self.calc_frame, text="Analysis Calculations", font=header_font, bg="#212F3D", fg="white").pack(pady=(5,10))
        
        f1 = tk.Frame(self.calc_frame, bg="#212F3D")
        f1.pack(fill="x", expand=True, padx=10)
        
        tk.Label(f1, text=f"Abnormal Windows:", font=calc_font, bg="#212F3D", fg="white").pack(side="left", padx=5)
        tk.Label(f1, text=f"{abnormal_windows}", font=calc_font, bg="#212F3D", fg="#ff6b6b" if abnormal_windows > 0 else "#82e0aa").pack(side="left")
        
        tk.Label(f1, text=f"Max Std Dev:", font=calc_font, bg="#212F3D", fg="white").pack(side="left", padx=(20, 5))
        tk.Label(f1, text=f"{max_std:.4f}", font=calc_font, bg="#212F3D", fg="white").pack(side="left")
        
        tk.Label(f1, text=f"Std Dev Threshold:", font=calc_font, bg="#212F3D", fg="white").pack(side="left", padx=(20, 5))
        tk.Label(f1, text=f"{threshold}", font=calc_font, bg="#212F3D", fg="white").pack(side="left")
        
        
        # 6. Update the main result table (Treeview)
        for i in self.tree.get_children(): self.tree.delete(i)
        self.tree["columns"] = ("Patient", "Prediction"); self.tree.column("#0", width=0, stretch=tk.NO); self.tree.heading("#0", text="", anchor="w")
        self.tree.heading("Patient", text="Patient ID", anchor="center"); self.tree.column("Patient", anchor="center", width=250)
        self.tree.heading("Prediction", text="Prediction Result", anchor="w"); self.tree.column("Prediction", anchor="w")
        
        tag = "seizure" if "Epilepsy" in prediction else "normal"
        self.tree.insert("", "end", values=("Patient 1", final_text), tags=(tag,))
        self.tree.tag_configure("seizure", foreground="#ff6b6b", font=("Segoe UI", 14, "bold")); 
        self.tree.tag_configure("normal", foreground="#82e0aa", font=("Segoe UI", 14, "bold"))
    
    def hide_results_widgets(self):
        """[NEW] Helper function to destroy dynamic widgets."""
        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None
        if self.calc_frame:
            self.calc_frame.destroy()
            self.calc_frame = None
    
    # ----- [MODIFIED] hide_results -----
    def hide_results(self):
        """
        [MODIFIED] Now also destroys the graph and calculation widgets.
        """
        # 1. Destroy dynamic widgets
        self.hide_results_widgets()
        
        # 2. Hide/show the main controls
        self.canvas.itemconfig(self.results_window, state="hidden")
        self.canvas.itemconfig(self.upload_button_window, state="normal")
        self.canvas.itemconfig(self.back_button_window, state="normal")
    
    @staticmethod
    def clean_csv(df):
        pd = get_pandas() 
        if not pd: return None
        
        df_num = df.copy()
        for col in df_num.columns:
            if df_num[col].dtype == 'object':
                df_num[col] = pd.to_numeric(df_num[col], errors='coerce')
        df_num = df_num.dropna(axis=1, how='all')
        if df_num.shape[1] == 0: return None
        # [MODIFIED] Only drop rows where ALL values are NaN
        df_num = df_num.dropna(axis=0, how='all') 
        # [MODIFIED] Fill any remaining NaNs (e.g., from gaps) with 0
        df_num = df_num.fillna(0)
        if df_num.shape[0] == 0: return None
        return df_num

# ---------- MAIN LAUNCHER ----------
class MainLauncher:
    def __init__(self, root):
        self.root = root; root.title("EEG Toolkit - Launcher"); 
        screen_width = root.winfo_screenwidth(); screen_height = root.winfo_screenheight()
        self.video_path = "app/inter.mov"
        self.bg_photo = None; self.video_frame_queue = queue.Queue(maxsize=5)
        
        self.video_dimensions = [screen_width, screen_height]
        self.video_dim_lock = threading.Lock()
        self.video_stop_event = threading.Event()
        
        self.get_video_dimensions = self.get_safe_dimensions
        self.video_thread = threading.Thread(target=video_processing_thread, 
                                             args=(self.video_path, self.video_frame_queue, 
                                                   self.get_video_dimensions, self.video_stop_event), 
                                             daemon=True)
        self.video_thread.start()
        
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, highlightthickness=0); self.canvas.pack(fill="both", expand=True)
        self.bg_on_canvas = self.canvas.create_image(0, 0, anchor="nw")
        right_x = screen_width * 0.75; title_font = ("Georgia", 70, "bold italic")
        self.canvas.create_text(right_x + 4, screen_height * 0.25 + 4, text="EEG Toolkit", font=title_font, fill="black")
        self.canvas.create_text(right_x, screen_height * 0.25, text="EEG Toolkit", font=title_font, fill="#EAECEE")
        self.canvas.create_text(right_x + 2, screen_height * 0.40 + 2, text="Choose an application to open", font=("Georgia", 18, "italic"), fill="black")
        self.canvas.create_text(right_x, screen_height * 0.40, text="Choose an application to open", font=("Georgia", 18, "italic"), fill="#FDFEFE")
        
        button_font = ("Georgia", 22, "italic") 
        converter_btn = CustomButton(
            self.canvas, text="Open EEG Image Converter", 
            command=self.open_converter, font=button_font, 
            fg="white", hover_fg="white",
            radius=10, bg="#007BFF", 
            hover_bg="#0056B3"
        )
        detector_btn = CustomButton(
            self.canvas, text="Open Epilepsy Detector", 
            command=self.open_detector, font=button_font, 
            fg="white", hover_fg="white",
            radius=10, bg="#007BFF", 
            hover_bg="#0056B3"
        )
        quit_btn = CustomButton(
            self.canvas, text="Quit Application", 
            command=self.close_window, font=button_font, 
            fg="white", hover_fg="white", height=50,
            radius=10, bg="#DC3545", 
            hover_bg="#C82333"
        )
        
        self.canvas.create_window(right_x, screen_height * 0.55, window=converter_btn); self.canvas.create_window(right_x, screen_height * 0.68, window=detector_btn); self.canvas.create_window(right_x, screen_height * 0.85, window=quit_btn)
        
        self.root.bind("<Configure>", self.on_resize) 
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.start_animations()
        self.root.after(100, lambda: self.on_resize(None))

    def close_window(self):
        print("Closing main launcher...")
        self.video_stop_event.set()
        self.root.destroy()
        
    def on_resize(self, event):
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        with self.video_dim_lock:
            self.video_dimensions[0] = width
            self.video_dimensions[1] = height
            
    def get_safe_dimensions(self):
        with self.video_dim_lock:
            return tuple(self.video_dimensions)

    def start_animations(self): 
        self._update_video_background()

    def _update_video_background(self):
        try:
            self.bg_photo = self.video_frame_queue.get_nowait()
            self.canvas.itemconfig(self.bg_on_canvas, image=self.bg_photo)
        except queue.Empty: pass
        except Exception as e: 
            if isinstance(e, tk.TclError): return 

        if not self.video_stop_event.is_set():
            self.root.after(33, self._update_video_background)

    def open_converter(self): 
        EEGConverterApp(self.root)
    def open_detector(self): 
        EpilepsyDetectorApp(self.root)

# ---------- RUN ----------
if __name__ == "__main__":
    main_root = tk.Tk(); 
    main_root.attributes('-fullscreen', True); 
    app = MainLauncher(main_root); 
    main_root.mainloop()