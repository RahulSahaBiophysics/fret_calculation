import sys
import napari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from magicgui import magicgui
from qtpy.QtWidgets import QApplication, QFileDialog
import os

# Ensure QApplication is created
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)
 
# Function to load multiple CSV files from a directory using Pandas
def load_csv_files(title):
    dir_path = QFileDialog.getExistingDirectory(None, title, "")
    if not dir_path:
        return None
    csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv")]
    if not csv_files:
        return None
    return [pd.read_csv(f).to_numpy() for f in csv_files]  # Using pandas to handle headers

# Load donor and acceptor CSV files
donor_files = load_csv_files("Select Donor CSV Directory")
acceptor_files = load_csv_files("Select Acceptor CSV Directory")
FRET_files = load_csv_files("Select FRET CSV Directory")

if donor_files is None or acceptor_files is None:
    print("Error: Could not load donor and/or acceptor CSV files.")
    sys.exit()

# Ensure all files have the same number of columns (frames)
num_frames = donor_files[0].shape[1]
'''if any(df.shape[1] != num_frames for df in donor_files + acceptor_files):
    print("Error: All donor and acceptor files must have the same number of frames.")
    sys.exit()'''

# Function to calculate FRET efficiency
def calculate_fret(donor, acceptor):
    return acceptor / (donor + acceptor)

# Directory for saving plots
save_folder = "plots"
os.makedirs(save_folder, exist_ok=True)

# Plot and save normalized intensity traces
for donor, fret, acceptor in zip(donor_files, acceptor_files, FRET_files):
    FRET_efficiency = calculate_fret(donor, fret)  # Compute FRET for all frames
    for i in range(donor.shape[0]):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Normalize traces
        donor_max = np.max(donor[i])
        acceptor_max = np.max(acceptor[i])
        fret_max = np.max(fret[i])
        norm_donor = donor[i] / donor_max
        norm_acceptor = acceptor[i] / acceptor_max
        norm_fret = fret [i] / fret_max
        
        # First subplot: Normalized intensity traces
        axes[0].plot(range(num_frames), norm_donor, label=f"Trace {i+1} Donor", color='green')
        axes[0].plot(range(num_frames), norm_acceptor, label=f"Trace {i+1} Acceptor", color='purple')
        axes[0].plot(range(num_frames), norm_fret, label=f"Trace {i+1} Acceptor", color='orange')
        axes[0].set_ylabel("Normalized Intensity")
        axes[0].set_title(f"Trace {i+1} Intensity & FRET Efficiency")
        axes[0].legend()
        
        # Second subplot: FRET efficiency
        axes[1].plot(range(num_frames), FRET_efficiency[i, :], label=f"Trace {i+1}", color='blue')
        axes[1].set_xlabel("Frames")
        axes[1].set_ylabel("FRET Efficiency")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"trace_{i+1}.png"))
        plt.close()


# Dictionary to store selected regions
selected_regions = {}

# Interactive widget for FRET selection
@magicgui(trace_index={"widget_type": "SpinBox", "min": 0, "max": len(donor_files[0]) - 1, "step": 1},
          start_frame={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1},
          end_frame={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1})
def select_fret_region(trace_index: int = 0, start_frame: int = 0, end_frame: int = 100):
    if end_frame <= start_frame:
        print("Invalid selection range!")
        return
    
    selected_regions[trace_index] = (start_frame, end_frame)
    print(f"Selected region for Trace {trace_index}: {start_frame} - {end_frame}")

# Function to plot after all regions are selected
def plot_selected_fret():
    if not selected_regions:
        print("No regions selected!")
        return
    
    fret_efficiency_list = []
    
    for trace_index, (start_frame, end_frame) in selected_regions.items():
        for donor, fret in zip(donor_files, FRET_files):
            donor_selected = donor[trace_index, start_frame:end_frame]
            acceptor_selected = fret[trace_index, start_frame:end_frame]
            
            mean_donor = np.mean(donor_selected)
            mean_acceptor = np.mean(acceptor_selected)
            
            fret_efficiency = calculate_fret(mean_donor, mean_acceptor)
            fret_efficiency_list.append(fret_efficiency)
    
    # Plot histogram of FRET efficiency values
    plt.figure(figsize=(8, 6))
    plt.hist(fret_efficiency_list, bins=20, color='blue', edgecolor='black')
    plt.xlabel("FRET Efficiency")
    plt.ylabel("Frequency")
    plt.title("Histogram of FRET Efficiency")
    plt.show()

# Napari viewer
viewer = napari.Viewer()
for donor, acceptor in zip(donor_files, acceptor_files):
    viewer.add_image(donor, name="Donor Intensity", colormap="Greens")
    viewer.add_image(acceptor, name="Acceptor Intensity", colormap="Reds")
viewer.window.add_dock_widget(select_fret_region, area="right")
viewer.window.add_dock_widget(magicgui(plot_selected_fret), area="right")
napari.run()