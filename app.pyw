
import tkinter as tk
from tkinter import Button, Label, Toplevel, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from matplotlib import pyplot as plt
import numpy as np
import scipy.fftpack
from tkinter import simpledialog
from collections import Counter



class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application")
        self.root.geometry("1200x700")
        self.root.state('zoomed')  # Start in full-screen mode

        # Initialize variables
        self.image = None
        self.original_image = None
        self.image_path = None
        self.active_button = None  # Track the active button

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Right frame for buttons
        button_frame = tk.Frame(self.root, bg="#2E3440")
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Canvas with scrollbar for buttons
        canvas = tk.Canvas(button_frame, bg="#2E3440", highlightthickness=0, width=240)
        scrollbar = tk.Scrollbar(button_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#2E3440")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # Enable mouse wheel scrolling
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda ev: self._on_mouse_wheel(ev, canvas)))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        self.button_style = {
            "font": ("Arial", 12, "bold"),
            "bg": "#5E81AC",
            "fg": "white",
            "activebackground": "#81A1C1",
            "activeforeground": "black",
            "width": 25,
            "pady": 5,
            "relief": tk.FLAT,
        }

        buttons = [
            ("Read Image", self.load_image),
            ("Save Image", self.save_image),
            ("Reset to Original", self.reset_to_original),  # Modified here
            ("Change RGB to Grayscale", self.convert_to_grayscale),
            ("Resize Image", self.resize_image),
            ("Rotate Image", self.rotate_image),
            ("Translate Image", self.translate_image),
            ("Thresholding", self.thresholding),
            ("Show Histogram (Gray)", self.show_histogram_gray),
            ("Show Histogram (RGB)", self.show_histogram_rgb),
            ("Histogram Equalization", self.histogram_equalization),
            ("Contrast Stretching", self.contrast_stretching),
            ("Negative Transformation", self.negative_transformation),
            ("Log Transformation", self.log_transformation),
            ("Power Law Transformation", self.power_law_transformation),
            ("Gaussian Blur", self.apply_gaussian_blur),
            ("Average Blur", self.apply_average_blur),
            ("Median Blur", self.apply_median_blur),
            ("Max Blur", self.apply_max_blur),
            ("Min Blur", self.apply_min_blur),
            ("Add Gaussian Noise", self.add_gaussian_noise),
            ("Add Salt & Pepper Noise", self.add_salt_pepper_noise),
            ("Sharpening Filters", self.sharpening_filters),
            ("Edge Detection (Sobel)", self.sobel_edge_detection),
            ("Edge Detection (Canny)", self.canny_edge_detection),
        ]

        self.buttons = []  # Store button widgets for styling

        for text, command in buttons:
            btn = tk.Button(
                scrollable_frame,
                text=text,
                command=lambda cmd=command, b=text: self.set_active_button(cmd, b),
                **self.button_style,
            )
            btn.pack(padx=5, pady=5, fill=tk.X)
            self.buttons.append(btn)

        # Canvas for displaying images
        self.original_canvas = tk.Canvas(self.root, bg="#ECEFF4", width=600, height=700)
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.modified_canvas = tk.Canvas(self.root, bg="#ECEFF4", width=600, height=700)
        self.modified_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _on_mouse_wheel(self, event, canvas):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def set_active_button(self, command, button_text):
        # Reset all button styles
        for btn in self.buttons:
            btn.config(bg="#5E81AC")  # Default color

        # Highlight the active button except for "Reset to Original"
        if button_text != "Reset to Original" and button_text != "Save Image" and button_text != "Read Image":
            for btn in self.buttons:
                if btn.cget("text") == button_text:
                    btn.config(bg="#A3BE8C")  # Active color

        # Execute the command
        command()

    # Image operations
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_path:
            return
        self.image_path = file_path
        self.image = cv2.imread(file_path)
        self.original_image = self.image.copy()
        self.display_image(self.original_image, self.original_canvas)
        self.clear_canvas(self.modified_canvas)

    def save_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if not file_path:
            return
        cv2.imwrite(file_path, self.image)
        messagebox.showinfo("Success", "Image saved successfully!")

    def reset_to_original(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return
        # Reset the image to the original one
        self.image = self.original_image.copy()
        self.display_image(self.image, self.original_canvas)
        # Clear the modified canvas
        self.clear_canvas(self.modified_canvas)

    def convert_to_grayscale(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.image, self.modified_canvas)

    # Placeholder for other functions

    def resize_image(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        resize_window = tk.Toplevel(self.root)
        resize_window.title("Resize Image")

        tk.Label(resize_window, text="Width:").grid(row=0, column=0, padx=10, pady=5)
        width_entry = tk.Entry(resize_window)
        width_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(resize_window, text="Height:").grid(row=1, column=0, padx=10, pady=5)
        height_entry = tk.Entry(resize_window)
        height_entry.grid(row=1, column=1, padx=10, pady=5)

        def apply_resize():
            try:
                width = int(width_entry.get())
                height = int(height_entry.get())
                resized_image = cv2.resize(self.original_image, (width, height))
                self.image = resized_image
                self.display_image(self.image, self.modified_canvas)
                resize_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integer dimensions.")

        apply_button = tk.Button(resize_window, text="Apply", command=apply_resize)
        apply_button.grid(row=2, column=0, columnspan=2, pady=10)



    def rotate_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window for rotating the image
        rotate_window = tk.Toplevel(self.root)
        rotate_window.title("Rotate Image")

        # Create a label and entry for the angle input
        tk.Label(rotate_window, text="Angle (degrees):").grid(row=0, column=0, padx=10, pady=5)
        angle_entry = tk.Entry(rotate_window)
        angle_entry.grid(row=0, column=1, padx=10, pady=5)

        def apply_rotation():
            try:
                angle = float(angle_entry.get())

                # Use the processed image if available, otherwise use the original image
                image_to_rotate =  self.original_image.copy()

                # Calculate the rotation matrix
                height, width = image_to_rotate.shape[:2]
                image_center = (width / 2, height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1)

                # Calculate the sine and cosine of the angle
                abs_cos = abs(rotation_matrix[0, 0])
                abs_sin = abs(rotation_matrix[0, 1])

                # Compute the new bounding dimensions of the image
                new_width = int(width * abs_cos + height * abs_sin)
                new_height = int(width * abs_sin + height * abs_cos)

                # Adjust the rotation matrix to take into account translation
                rotation_matrix[0, 2] += (new_width / 2) - image_center[0]
                rotation_matrix[1, 2] += (new_height / 2) - image_center[1]

                # Perform the actual rotation
                rotated_image = cv2.warpAffine(image_to_rotate, rotation_matrix, (new_width, new_height))

                # Display the rotated image
                self.display_image(rotated_image, self.modified_canvas)
                self.image = rotated_image

                rotate_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid angle.")

        # Apply button to apply the rotation
        apply_button = tk.Button(rotate_window, text="Apply", command=apply_rotation)
        apply_button.grid(row=1, column=0, columnspan=2, pady=10)




    def translate_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # First window to get the matrix dimensions
        dimensions_window = tk.Toplevel(self.root)
        dimensions_window.title("Translation Matrix Dimensions")

        tk.Label(dimensions_window, text="Number of Rows:").grid(row=0, column=0, padx=10, pady=5)
        rows_entry = tk.Entry(dimensions_window)
        rows_entry.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(dimensions_window, text="Number of Columns:").grid(row=1, column=0, padx=10, pady=5)
        cols_entry = tk.Entry(dimensions_window)
        cols_entry.grid(row=1, column=1, padx=10, pady=5)

        def proceed_to_matrix_input():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())

                if rows != 2 or cols != 3:
                    messagebox.showerror("Error", "Matrix must be 2x3 for affine transformation.")
                    return

                dimensions_window.destroy()

                # Second window to input matrix values
                matrix_window = tk.Toplevel(self.root)
                matrix_window.title("Enter Translation Matrix Values")

                matrix_entries = []

                for i in range(rows):
                    row_entries = []
                    for j in range(cols):
                        entry = tk.Entry(matrix_window, width=10)
                        entry.grid(row=i, column=j, padx=5, pady=5)
                        row_entries.append(entry)
                    matrix_entries.append(row_entries)

                def apply_translation():
                    try:
                        # Read matrix values from entries
                        matrix_values = [
                            [float(matrix_entries[i][j].get()) for j in range(cols)]
                            for i in range(rows)
                        ]
                        
                        # Construct the translation matrix for cv2.warpAffine
                        translation_matrix = np.float32(matrix_values)

                        # Apply the transformation using cv2.warpAffine
                        translated_image = cv2.warpAffine(self.original_image, translation_matrix, 
                        (self.original_image.shape[1], self.original_image.shape[0]))

                        # Show the translated image
                        self.image = translated_image
                        self.display_image(self.image, self.modified_canvas)
                        matrix_window.destroy()
                    except ValueError:
                        messagebox.showerror("Error", "Please ensure all inputs are valid numbers.")

                tk.Button(matrix_window, text="Apply", command=apply_translation).grid(row=rows, column=0, columnspan=cols, pady=10)

            except ValueError:
                messagebox.showerror("Error", "Please enter valid dimensions.")

        tk.Button(dimensions_window, text="Next", command=proceed_to_matrix_input).grid(row=2, column=0, columnspan=2, pady=10)


    def show_histogram_gray(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
    
    # Convert the original image to grayscale
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram
        histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Plot the histogram using matplotlib
        plt.figure("Gray Histogram")
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.plot(histogram, color='gray')
        plt.xlim([0, 256])
        plt.grid(True)
        plt.show()

    def show_histogram_rgb(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        # Split the original image into its Red, Green, and Blue channels
        channels = cv2.split(self.original_image)
        colors = ('b', 'g', 'r')  # Corresponding colors for the histogram
        labels = ('Blue', 'Green', 'Red')

        # Plot the histograms for each channel
        plt.figure("RGB Histogram")
        plt.title("RGB Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        for channel, color, label in zip(channels, colors, labels):
            # Calculate the histogram for the channel
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            # Plot the histogram
            plt.plot(histogram, color=color, label=label)

        plt.xlim([0, 256])
        plt.legend()  # Add legend to differentiate channels
        plt.grid(True)
        plt.show()



    def histogram_equalization(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Convert the original image to YCrCb color space
        ycrcb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YCrCb)
    
    # Split the channels
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
    
    # Equalize the Y channel (luminance)
        equalized_y_channel = cv2.equalizeHist(y_channel)
    
    # Merge the channels back
        equalized_image = cv2.merge((equalized_y_channel, cr_channel, cb_channel))
    
    # Convert back to BGR color space
        equalized_image_bgr = cv2.cvtColor(equalized_image, cv2.COLOR_YCrCb2BGR)
    
    # Update the displayed image
        self.image = equalized_image_bgr
        self.display_image(self.image, self.modified_canvas)
    



    def contrast_stretching(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window to get input min and max pixel values
        stretch_window = tk.Toplevel(self.root)
        stretch_window.title("Contrast Stretching Parameters")

        # Labels and input entries for min and max values
        tk.Label(stretch_window, text="Min Intensity (default: 0):").grid(row=0, column=0, padx=10, pady=5)
        min_entry = tk.Entry(stretch_window)
        min_entry.grid(row=0, column=1, padx=10, pady=5)
        min_entry.insert(0, "0")  # Default value

        tk.Label(stretch_window, text="Max Intensity (default: 255):").grid(row=1, column=0, padx=10, pady=5)
        max_entry = tk.Entry(stretch_window)
        max_entry.grid(row=1, column=1, padx=10, pady=5)
        max_entry.insert(0, "255")  # Default value

        def apply_contrast_stretch():
            try:
                # Parse input values
                min_intensity = int(min_entry.get())
                max_intensity = int(max_entry.get())

                if min_intensity < 0 or max_intensity > 255 or min_intensity >= max_intensity:
                    messagebox.showerror("Error", "Invalid intensity range. Must be between 0 and 255, and Min < Max.")
                    return

                # Perform contrast stretching
                img_float = self.original_image.astype(np.float32)

                # Calculate the min and max pixel intensities for stretching
                img_min = np.min(img_float)
                img_max = np.max(img_float)

                # Stretch the image
                stretched_image = ((img_float - img_min) / (img_max - img_min)) * (max_intensity - min_intensity) + min_intensity
                stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

                # Update the displayed image
                self.image = stretched_image
                self.display_image(self.image, self.modified_canvas)
                stretch_window.destroy()

            except ValueError:
                messagebox.showerror("Error", "Please enter valid numerical values for intensities.")

        # Apply button to apply contrast stretching
        apply_button = tk.Button(stretch_window, text="Apply", command=apply_contrast_stretch)
        apply_button.grid(row=2, column=0, columnspan=2, pady=10)


    def thresholding(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Function to apply thresholding
        def apply_threshold(threshold_value):
            # Convert to grayscale if necessary
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Convert back to BGR for consistent display
            self.image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image, self.modified_canvas)

        # Function for automatic thresholding (Otsu's method)
        def automatic_thresholding():
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Otsu's thresholding
            _, otsu_thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to BGR for consistent display
            self.image = cv2.cvtColor(otsu_thresholded_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.image, self.modified_canvas)

        # Create an initial window for thresholding mode selection
        mode_window = tk.Toplevel(self.root)
        mode_window.title("Thresholding Mode")

        tk.Label(mode_window, text="Select Thresholding Mode:", font=("Arial", 12)).pack(pady=10)

        # Manual thresholding button
        def open_manual_thresholding():
            mode_window.destroy()

            # Create another window to enter threshold value
            manual_window = tk.Toplevel(self.root)
            manual_window.title("Manual Thresholding")

            tk.Label(manual_window, text="Enter Threshold Value (0-255):").grid(row=0, column=0, padx=10, pady=5)
            threshold_entry = tk.Entry(manual_window)
            threshold_entry.grid(row=0, column=1, padx=10, pady=5)

            def apply_manual_threshold():
                try:
                    threshold_value = int(threshold_entry.get())
                    if 0 <= threshold_value <= 255:
                        apply_threshold(threshold_value)
                        manual_window.destroy()
                    else:
                        messagebox.showerror("Error", "Threshold value must be between 0 and 255.")
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer threshold value.")

            # Apply button
            tk.Button(manual_window, text="Apply", command=apply_manual_threshold).grid(row=1, column=0, columnspan=2, pady=10)

        # Buttons for selecting mode
        tk.Button(mode_window, text="Manual Thresholding", width=25, command=open_manual_thresholding).pack(pady=5)
        tk.Button(mode_window, text="Automatic Thresholding (Otsu)", width=25, command=lambda: [automatic_thresholding(), mode_window.destroy()]).pack(pady=5)


    def negative_transformation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Perform negative transformation
        negative_image = 255 - self.original_image

        # Update the image and display the result
        self.image = negative_image
        self.display_image(self.image, self.modified_canvas)


    def log_transformation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Convert the original image to a float32 type for computation
        image_float = self.original_image.astype(np.float32)

        # Perform log transformation: s = c * log(1 + r)
        c = 255 / np.log(1 + np.max(image_float))  # Scaling constant
        log_image = c * np.log(1 + image_float)

        # Clip the values to the valid range [0, 255] and convert back to uint8
        log_image = np.clip(log_image, 0, 255).astype(np.uint8)

        # Update the image and display the result
        self.image = log_image
        self.display_image(self.image, self.modified_canvas)




    def power_law_transformation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window to input the gamma value
        gamma_window = tk.Toplevel(self.root)
        gamma_window.title("Power-Law Transformation (Gamma Correction)")

        # Label and Entry for gamma input
        tk.Label(gamma_window, text="Enter Gamma Value (e.g., 0.5 for brightening, >1 for darkening):").grid(row=0, column=0, padx=10, pady=5)
        gamma_entry = tk.Entry(gamma_window)
        gamma_entry.grid(row=0, column=1, padx=10, pady=5)

        def apply_gamma():
            try:
                gamma = float(gamma_entry.get())
                if gamma <= 0:
                    messagebox.showerror("Error", "Gamma value must be greater than 0.")
                    return

                # Normalize the image to the range [0, 1]
                normalized_image = self.original_image.astype(np.float32) / 255.0

                # Apply the power-law transformation
                gamma_corrected_image = np.power(normalized_image, gamma)

                # Scale back to [0, 255] and convert to uint8
                gamma_corrected_image = (gamma_corrected_image * 255).clip(0, 255).astype(np.uint8)

                # Update the image and display the result
                self.image = gamma_corrected_image
                self.display_image(self.image, self.modified_canvas)

                # Close the gamma input window
                gamma_window.destroy()

            except ValueError:
                messagebox.showerror("Error", "Please enter a valid numerical gamma value.")

        # Apply button
        apply_button = tk.Button(gamma_window, text="Apply", command=apply_gamma)
        apply_button.grid(row=1, column=0, columnspan=2, pady=10)


    def apply_gaussian_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Gaussian Blur Parameters")

        # Labels and entries for kernel size (odd numbers only)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 3, 5, 7):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "5")  # Default kernel size is 5

        def apply_blur():
            try:
                # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

                # Apply Gaussian blur
                blurred_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

                # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

        # Apply button to perform Gaussian blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)


    def apply_average_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Average Blur Parameters")

        # Labels and entries for kernel size (odd numbers only)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 3, 5, 7):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "5")  # Default kernel size is 5

        def apply_blur():
            try:
                # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

                # Apply Average blur
                blurred_image = cv2.blur(self.original_image, (kernel_size, kernel_size))

                # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

        # Apply button to perform average blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)


    def apply_median_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Median Blur Parameters")

        # Label and entry for kernel size (odd numbers only)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 3, 5, 7):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "5")  # Default kernel size is 5

        def apply_blur():
            try:
                # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

                # Apply Median blur
                blurred_image = cv2.medianBlur(self.original_image, kernel_size)

                # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

        # Apply button to perform median blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)


    def apply_max_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Max Blur Parameters")

    # Label and entry for kernel size (must be odd numbers)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 31, 51, 101):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "51")  # Default kernel size is 51

        def apply_blur():
            try:
            # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

            # Apply Gaussian Blur with a very large kernel
                blurred_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

            # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

    # Apply button to perform max blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)


    def apply_min_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Min Blur Parameters")

    # Label and entry for kernel size (must be odd numbers)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 3, 5, 7):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "3")  # Default kernel size is 3

        def apply_blur():
            try:
            # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

            # Apply Gaussian Blur with a very small kernel
                blurred_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

            # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

    # Apply button to perform min blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)

    def apply_max_blur(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get the kernel size
        blur_window = tk.Toplevel(self.root)
        blur_window.title("Max Blur Parameters")

    # Label and entry for kernel size (must be odd numbers)
        tk.Label(blur_window, text="Kernel Size (Odd Number, e.g., 31, 51, 101):").grid(row=0, column=0, padx=10, pady=5)
        kernel_entry = tk.Entry(blur_window)
        kernel_entry.grid(row=0, column=1, padx=10, pady=5)
        kernel_entry.insert(0, "51")  # Default kernel size is 51

        def apply_blur():
            try:
            # Parse and validate the kernel size
                kernel_size = int(kernel_entry.get())
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    messagebox.showerror("Error", "Kernel size must be a positive odd number.")
                    return

            # Apply Gaussian Blur with a very large kernel
                blurred_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)

            # Update the image and display the result
                self.image = blurred_image
                self.display_image(self.image, self.modified_canvas)

                blur_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid integer for kernel size.")

    # Apply button to perform max blur
        tk.Button(blur_window, text="Apply", command=apply_blur).grid(row=1, column=0, columnspan=2, pady=10)


    def add_gaussian_noise(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get the noise parameters
        noise_window = tk.Toplevel(self.root)
        noise_window.title("Gaussian Noise Parameters")

    # Label and entry for mean and standard deviation
        tk.Label(noise_window, text="Mean (e.g., 0):").grid(row=0, column=0, padx=10, pady=5)
        mean_entry = tk.Entry(noise_window)
        mean_entry.grid(row=0, column=1, padx=10, pady=5)
        mean_entry.insert(0, "0")

        tk.Label(noise_window, text="Standard Deviation (e.g., 25):").grid(row=1, column=0, padx=10, pady=5)
        std_dev_entry = tk.Entry(noise_window)
        std_dev_entry.grid(row=1, column=1, padx=10, pady=5)
        std_dev_entry.insert(0, "25")

        def apply_noise():
            try:
            # Parse and validate the mean and standard deviation
                mean = float(mean_entry.get())
                std_dev = float(std_dev_entry.get())

            # Generate Gaussian noise
                noise = np.random.normal(mean, std_dev, self.original_image.shape).astype(np.float32)

            # Add noise to the original image
                noisy_image = cv2.add(self.original_image.astype(np.float32), noise)
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

            # Update the image and display the result
                self.image = noisy_image
                self.display_image(self.image, self.modified_canvas)

                noise_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for mean and standard deviation.")

    # Apply button to add Gaussian noise
        tk.Button(noise_window, text="Apply", command=apply_noise).grid(row=2, column=0, columnspan=2, pady=10)

    def add_salt_pepper_noise(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get the noise parameters
        noise_window = tk.Toplevel(self.root)
        noise_window.title("Salt and Pepper Noise Parameters")

    # Label and entry for noise ratio
        tk.Label(noise_window, text="Noise Ratio (e.g., 0.05 for 5%):").grid(row=0, column=0, padx=10, pady=5)
        ratio_entry = tk.Entry(noise_window)
        ratio_entry.grid(row=0, column=1, padx=10, pady=5)
        ratio_entry.insert(0, "0.05")

        def apply_noise():
            try:
            # Parse and validate the noise ratio
                noise_ratio = float(ratio_entry.get())
                if not (0 <= noise_ratio <= 1):
                    messagebox.showerror("Error", "Noise ratio must be between 0 and 1.")
                    return

            # Generate salt and pepper noise
                noisy_image = self.original_image.copy()
                num_salt = int(noise_ratio * noisy_image.size * 0.5)
                num_pepper = int(noise_ratio * noisy_image.size * 0.5)

            # Add salt noise (white pixels)
                coords = [
                    np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape[:2]
                ]
                noisy_image[coords[0], coords[1]] = 255

            # Add pepper noise (black pixels)
                coords = [
                    np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape[:2]
                ]
                noisy_image[coords[0], coords[1]] = 0

            # Update the image and display the result
                self.image = noisy_image
                self.display_image(self.image, self.modified_canvas)

                noise_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid number for noise ratio.")

    # Apply button to add salt and pepper noise
        tk.Button(noise_window, text="Apply", command=apply_noise).grid(row=1, column=0, columnspan=2, pady=10)

    def sharpening_filters(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get kernel dimensions and values
        sharpen_window = tk.Toplevel(self.root)
        sharpen_window.title("Sharpening Kernel Parameters")

    # Label and entry for kernel rows
        tk.Label(sharpen_window, text="Kernel Rows:").grid(row=0, column=0, padx=10, pady=5)
        rows_entry = tk.Entry(sharpen_window)
        rows_entry.grid(row=0, column=1, padx=10, pady=5)
        rows_entry.insert(0, "3")  # Default rows

    # Label and entry for kernel columns
        tk.Label(sharpen_window, text="Kernel Columns:").grid(row=1, column=0, padx=10, pady=5)
        cols_entry = tk.Entry(sharpen_window)
        cols_entry.grid(row=1, column=1, padx=10, pady=5)
        cols_entry.insert(0, "3")  # Default columns

        def apply_custom_kernel():
            try:
            # Parse kernel dimensions
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())

                if rows <= 0 or cols <= 0:
                    messagebox.showerror("Error", "Kernel dimensions must be positive integers.")
                    return

            # Create a new window to input kernel values
                kernel_window = tk.Toplevel(self.root)
                kernel_window.title("Kernel Values")

            # Create entry fields for each kernel value
                kernel_entries = []
                for i in range(rows):
                    row_entries = []
                    for j in range(cols):
                        entry = tk.Entry(kernel_window, width=5)
                        entry.grid(row=i, column=j, padx=5, pady=5)
                        row_entries.append(entry)
                    kernel_entries.append(row_entries)

                def apply_kernel():
                    try:
                    # Read kernel values from entries
                        kernel = np.zeros((rows, cols), dtype=float)
                        for i in range(rows):
                            for j in range(cols):
                                kernel[i, j] = float(kernel_entries[i][j].get())

                    # Apply sharpening filter
                        sharpened_image = cv2.filter2D(self.original_image, -1, kernel)

                    # Update the image and display the result
                        self.image = sharpened_image
                        self.display_image(self.image, self.modified_canvas)

                        kernel_window.destroy()
                        sharpen_window.destroy()
                    except ValueError:
                        messagebox.showerror("Error", "Please enter valid numbers for all kernel values.")

            # Apply button to process kernel
                tk.Button(kernel_window, text="Apply", command=apply_kernel).grid(row=rows, column=0, columnspan=cols, pady=10)

            except ValueError:
                messagebox.showerror("Error", "Please enter valid integers for kernel dimensions.")

    # Next button to define kernel values
        tk.Button(sharpen_window, text="Next", command=apply_custom_kernel).grid(row=2, column=0, columnspan=2, pady=10)


    def sobel_edge_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Convert image to grayscale if it isn't already
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Combine the gradients
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # Update the image and display the result
        self.image = sobel_combined
        self.display_image(self.image, self.modified_canvas)


    def canny_edge_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

    # Create a new window to get Canny parameters
        canny_window = tk.Toplevel(self.root)
        canny_window.title("Canny Edge Detection Parameters")

    # Label and entry for lower threshold
        tk.Label(canny_window, text="Lower Threshold:").grid(row=0, column=0, padx=10, pady=5)
        lower_thresh_entry = tk.Entry(canny_window)
        lower_thresh_entry.grid(row=0, column=1, padx=10, pady=5)
        lower_thresh_entry.insert(0, "50")

    # Label and entry for upper threshold
        tk.Label(canny_window, text="Upper Threshold:").grid(row=1, column=0, padx=10, pady=5)
        upper_thresh_entry = tk.Entry(canny_window)
        upper_thresh_entry.grid(row=1, column=1, padx=10, pady=5)
        upper_thresh_entry.insert(0, "150")

        def apply_canny():
            try:
            # Parse thresholds
                lower_thresh = int(lower_thresh_entry.get())
                upper_thresh = int(upper_thresh_entry.get())

            # Apply Canny edge detection
                edges = cv2.Canny(self.original_image, lower_thresh, upper_thresh)

            # Update the image and display the result
                self.image = edges
                self.display_image(self.image, self.modified_canvas)

                canny_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid integers for thresholds.")

    # Apply button to perform Canny edge detection
        tk.Button(canny_window, text="Apply", command=apply_canny).grid(row=2, column=0, columnspan=2, pady=10)


    def display_image(self, img, canvas):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        img_tk = ImageTk.PhotoImage(img)
        canvas.image = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    def clear_canvas(self, canvas):
        canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()




