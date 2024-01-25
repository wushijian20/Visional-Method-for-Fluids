import tkinter as tk
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
# Import packages
import cv2 as cv
import numpy as np
import pandas as pd
import tkinter.font as tkFont
from matplotlib import pyplot as plt 
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter 
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os



##################################################################
# helper function to trim trailing zeros
def trim_trailing_zeros(series):
    for i in reversed(range(len(series))):
        if series.iat[i] != 0:   # !=
            return series.iloc[:i+1]
    return pd.Series(dtype='float64')

# Function to remove last n non-NaN values
def remove_last_n(df, n):
    for col in df.columns:
        # Get non-NA indices
        non_na_indices = df[col].dropna().index
        
        # If there are less than 'n' non-NAs, continue
        if len(non_na_indices) < n:
            continue

        # Get last 'n' non-NA indices
        last_n_indices = non_na_indices[-n:]

        # Set last 'n' non-NAs to NaN
        df.loc[last_n_indices, col] = np.nan

    return df

def find_local_peak(series, threshold):
    for i in range(1, len(series)-1):  # Skip first and last element
        if series[i] > threshold and series[i-1] < series[i] > series[i+1]:
            return series.index[i]
    return np.nan  # Return NaN if no peak is found


def save_to_csv(sample_name, rupture_time):
    filename = 'results/results.csv'
    data = {'Sample Name': [sample_name], 'Rupture Time': [rupture_time]}
    df = pd.DataFrame(data)

    # Check if file exists. If not, write header, otherwise append without header
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


##################################################################

root = tk.Tk()

# Set up the canvas
canvas = tk.Canvas(root, width=1000, height=600)
canvas.grid(columnspan=8, rowspan=8)

# Logo (Adjust the path to your logo image)
logo = Image.open('fluids.jpg')
# Resize the logo (e.g., width=200, height=200)
logo = logo.resize((150, 150))
logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image=logo)
logo_label.image = logo
logo_label.grid(columnspan=1,column=5, row=0)

# Instructions
# instructions = tk.Label(root, text="Select a video", font="Raleway")
# instructions.grid(columnspan=1, column=5, row=1)

# Instructions
instructions = tk.Label(root, text="The Break Time is:", font="Raleway")
instructions.grid(columnspan=1, column=5, row=4)

# Text Widget to Display Rupture Time
customFont = tkFont.Font(family="Helvetica", size=15, weight="bold")

rupture_time_text = tk.Text(root, height=2, width=15, font=customFont)
rupture_time_text.grid(columnspan=1, column=5, row=5)

# Label for sample name input
sample_name_label = tk.Label(root, text="Input the Sample Name:", font="Raleway")

sample_name_label.grid(column=5, row=1)

# Entry widget for sample name
sample_name_entry = tk.Entry(root, font="Raleway")
sample_name_entry.grid(column=5, row=2)



# Function to handle video opening
def open_file():
    browse_text.set("Loading...")
    file_path = askopenfile(mode='r', title="Choose a file", filetypes=[("AVI video file", "*.avi")])
    if file_path:
        browse_text.set("Select a Video")
        filament_diameter = []
        # play_video(file_path.name)
        # Call play_video and get rupture time
        rupture_time = play_video(file_path.name)

        # Get sample name from entry widget
        sample_name = sample_name_entry.get()

        # Save the data to CSV
        save_to_csv(sample_name, rupture_time)

        # Update GUI with rupture time
        rupture_time_text.delete(1.0, tk.END)
        rupture_time_text.insert(tk.END, f"{rupture_time} sec")


def play_video(file_path):
    filament_diameter = []
    dfs = []
    cap = cv2.VideoCapture(file_path)

    # frame_delay = 1 / 5  # 5 frames per second
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = np.rot90(frame)
        # Convert frame to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 10, 100)

        #################################################
        rows, cols = len(edges), len(edges[0])
        # print(edges)
        # print((rows, cols))

        left_edges, right_edges = [0]*rows, [0]*rows
        
        for i in range(0, rows, 1):
            if np.all(edges[i] == 0):
                left_edges[i] = 0

            for j in range(cols):
                if edges[i][j] == 255:
                    left_edges[i] = j
                    break
    

        for i in range(0, rows, 1):
            if np.all(edges[i] == 0):
                right_edges[i] =0
            for j in range(cols-1, -1, -1):
                if edges[i][j] == 255:
                    right_edges[i] = j
                    break

        filament_diameter.append(np.subtract(right_edges, left_edges))

        df_filament_diameter = pd.DataFrame(filament_diameter).T
        df_filament_diameter.shape
        # print(df_filament_diameter.shape)

        #################################################

        # Convert the original and edges frames to RGB for displaying in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # Convert to PIL format
        frame_pil = Image.fromarray(frame_rgb)
        edges_pil = Image.fromarray(edges_rgb)

        # Convert to ImageTk format
        frame_imgtk = ImageTk.PhotoImage(image=frame_pil)
        edges_imgtk = ImageTk.PhotoImage(image=edges_pil)

        # Display the original and edge-detected images
        canvas.create_image(20, 20, anchor=tk.NW, image=frame_imgtk)
        canvas.create_image(220, 20, anchor=tk.NW, image=edges_imgtk)  # Adjust position as needed

        # This is necessary to update the GUI
        root.update_idletasks()
        root.update()

    cap.release()

    # print(df_filament_diameter.shape)
    df1 = df_filament_diameter
    df2 = df1.apply(trim_trailing_zeros)
    df2 = df2.rolling(window=5, min_periods=1).mean()
    # Remove the last 2 non-NAs (Replace 2 with 100 for your case)
    df3 = remove_last_n(df2, 100)
    df3 = df2
    min_values = df3.min()
    df_min_values = min_values.to_frame()
    df_smoothed_min_values = df_min_values.rolling(window=3).mean()

    # Calculate the derivative (difference)
    df_diff = abs(df_smoothed_min_values.diff())
    # Find first local peak larger than 3 for all columns
    peak_indices = df_diff.apply(find_local_peak, args=(0.2,), axis=0)

    rupture_time = (df1.shape[1]-np.int64(peak_indices))/200
    rupture_time = rupture_time[0] 

    # Update the text widget with the rupture time
    rupture_time_text.delete(1.0, tk.END)  # Clear existing text
    rupture_time_text.insert(tk.END, f"{rupture_time} sec")  # Insert new text

    print("Rupture Time is:", (df1.shape[1]-np.int64(peak_indices))/200)
    return rupture_time



# Browse button
browse_text = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browse_text, command=open_file, font="Raleway", bg="#20bebe", fg="white", height=2, width=15)
browse_text.set("Select a Video")
browse_btn.grid(column=5, row=3)

root.mainloop()
