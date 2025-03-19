import sys
import napari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from magicgui import magicgui
from qtpy.QtWidgets import QApplication, QFileDialog
import os
import json


viewer = napari.Viewer()

# Ensure QApplication is created
app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)
 
# Function to load multiple CSV files from a directory using Pandas
def load_csv_files(title):
    dir_path = QFileDialog.getExistingDirectory(None, title, "")
    if not dir_path:
        return None
    csv_files = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv")])
    if not csv_files:
        return None
    
    data = []
    for f in csv_files:
        # Load the CSV, skipping the header row (header=0 skips the first row, which is the header)
        df = pd.read_csv(f, header=0)
        data.append(df)
    return data
    
    #return [pd.read_csv(f, header=0).to_numpy() for f in csv_files]  # Using pandas to handle headers


# Load donor and acceptor CSV files
donor_files = load_csv_files("Select Donor CSV Directory")
acceptor_files = load_csv_files("Select Acceptor CSV Directory")
FRET_files = load_csv_files("Select FRET CSV Directory")

print(f"Donor files: {len(donor_files)}")
for i, donor in enumerate(donor_files):
    print(f"Donor file {i} shape: {donor.shape}")

print(f"Acceptor files: {len(acceptor_files)}")
for i, acceptor in enumerate(acceptor_files):
    print(f"Acceptor file {i} shape: {acceptor.shape}")

print(f"FRET files: {len(FRET_files)}")
for i, FRET in enumerate(FRET_files):
    print(f"FRET file {i} shape: {FRET.shape}")


if donor_files is None or acceptor_files is None:
    print("Error: Could not load donor and/or acceptor CSV files.")
    sys.exit()

print(donor_files[0].shape)

num_frames = donor_files[0].shape[1]
print("number of frames", num_frames)


# Function to calculate FRET efficiency
def calculate_fret(donor, acceptor):
    return acceptor / (donor + acceptor)

def correct_acceptor (donor, acceptor , fret, alpha, beta):
    return fret - alpha*donor - beta*acceptor

def calculate_corrected_fret(donor, acceptor , fret, alpha, beta,gamma):
    return (fret - alpha*donor - beta*acceptor)/ ((gamma* donor)+fret - alpha*donor - beta*acceptor)

# Choose directory for saving plots and data
save_folder = QFileDialog.getExistingDirectory(None, "Select Directory to Save Plots and Data", "")
if not save_folder:
    print("No directory selected. Exiting.")
    sys.exit()

os.makedirs(save_folder, exist_ok=True)

# Plot and save normalized intensity traces
trace_count = 0  # Initialize the trace counter for all files (no reset between files)
for file_id, (donor_file, acceptor_file, fret_file) in enumerate(zip(donor_files, acceptor_files, FRET_files), start=1):

    donor = donor_file.to_numpy()
    acceptor = acceptor_file.to_numpy()
    fret = fret_file.to_numpy()

    # Iterate through each trace (row) in the current file
    for i in range(donor.shape[0]): 
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Normalize the intensity
        norm_acceptor = acceptor[i, :] #- np.min(acceptor[i, :])) / (np.max(acceptor[i, :]) - np.min(acceptor[i, :]))
        norm_fret = fret[i, :] #- np.min(fret[i, :])) / (np.max(fret[i, :]) - np.min(fret[i, :]))
        norm_donor = donor[i, :] #- np.min(donor[i, :])) / (np.max(donor[i, :]) - np.min(donor[i, :]))

        # First subplot: Normalized intensity traces
        axes[0].plot(range(num_frames), norm_donor, label=f"File {file_id} Trace {i} Donor", color='green')
        axes[0].plot(range(num_frames), norm_acceptor, label=f"File {file_id} Trace {i} Acceptor(direct excitation)", color='purple')
        axes[0].plot(range(num_frames), norm_fret, label=f"File {file_id} Trace {i} Acceptor(FRET)", color='orange')
        axes[0].set_ylabel("Normalized Intensity")
        axes[0].set_title(f"File {file_id} - Trace {i} Intensity & FRET Efficiency")
        axes[0].legend()

        # Second subplot: FRET efficiency
        FRET_efficiency = calculate_fret(donor[i, :], fret[i, :])
        axes[1].plot(range(num_frames), FRET_efficiency, label=f"File {file_id} Trace {i} FRET efficiency", color='black')
        axes[1].set_xlabel("Frames")
        axes[1].set_ylabel("FRET Efficiency")
        axes[1].set_ylim(0.3,0.8)
        axes[1].legend()

        axes[1].set_xticks(np.arange(0,1001,50))

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"trace_{trace_count+1}.png"))
        plt.close()

        trace_count += 1  # Increment after every trace processed



    # Dictionary to store selected regions
    selected_regions = {}

    # Interactive widget for FRET selection
    @magicgui(
        file_index={"widget_type": "SpinBox", "min": 0, "max": len(donor_files) - 1, "step": 1},
        trace_index={"widget_type": "SpinBox", "min": 0, "max": 0, "step": 1},
        donor_start={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1},
        donor_end={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1},
        acceptor_start={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1},
        acceptor_end={"widget_type": "Slider", "min": 0, "max": num_frames - 1, "step": 1},
    )
    def select_fret_region(
        file_index: int = 0,
        trace_index: int = 0,
        donor_start: int = 0,
        donor_end: int = 0,
        acceptor_start: int = 0,
        acceptor_end: int = 0
    ):
        global selected_regions
        if file_index >= len(donor_files):
            print("invalid file index!")
            return
        
        max_traces = donor_files[file_index].shape[0]-1
        select_fret_region.trace_index.max = max_traces

        if trace_index > max_traces:
            trace_index = max_traces
        select_fret_region.trace_index.value = trace_index
        
        if file_index not in selected_regions:
            selected_regions[file_index] = {}

        '''if file_index not in selected_regions:
            start_frame = 0
            end_frame = num_frames - 1'''

        if donor_end<= donor_start or acceptor_end<= acceptor_start:
            print("Invalid selection range!")
            return
        
        FRET_start = max(donor_start, acceptor_start)
        FRET_end = min(donor_end, acceptor_end)
    
        
        selected_regions[file_index][trace_index] = (donor_start, donor_end, acceptor_start, acceptor_end, FRET_start, FRET_end)
        print(f"Selected region for File {file_index}, Trace {trace_index}: {FRET_start} - {FRET_end}")

        select_fret_region.donor_start.value = FRET_start
        select_fret_region.donor_end.value = FRET_end
        select_fret_region.acceptor_start.value = FRET_start
        select_fret_region.acceptor_end.value = FRET_end

    @select_fret_region.file_index.changed.connect
    def update_trace_index(file_index):
        max_traces = donor_files[file_index].shape[0] -1
        select_fret_region.trace_index.max = max_traces
        if select_fret_region.trace_index.value > max_traces:
            select_fret_region.trace_index.value = max_traces


    mean_alpha_factor = None
    mean_beta_factor = None
    mean_gamma_factor = None

    @magicgui(call_button= "Plot Apparent FRET and alpha and beta")

    # Function to plot after all regions are selected
    def plot_apparent_fret():
        global mean_alpha_factor, mean_beta_factor
        if not selected_regions:
            print("No regions selected!")
            return
    
        fret_efficiency_list_apparent = []
        frame_wise_fret = []
        mean_fret_list = []
        alpha_factor_list = []
        beta_factor_list = []
        
        for file_index, traces in selected_regions.items():
            donor_file_2 = donor_files[file_index]
            acceptor_file_2 = acceptor_files[file_index]
            fret_file_2 = FRET_files[file_index]

            donor_2 = donor_file_2.to_numpy()
            acceptor_2 = acceptor_file_2.to_numpy()
            fret_2 = fret_file_2.to_numpy()
                
            for trace_index, (donor_start, donor_end, acceptor_start, acceptor_end, FRET_start, FRET_end) in traces.items():    
                    print(donor_2.shape)
                    print("trace index", trace_index)

                    donor_selected = donor_2[trace_index, FRET_start:FRET_end]
                    acceptor_selected = fret_2[trace_index, FRET_start:FRET_end]

                    fret_values = [calculate_fret(d, a) for d, a in zip(donor_selected, acceptor_selected)]
                    fret_efficiency_list_apparent.extend(fret_values)
                    frame_wise_fret.append(fret_values)

                    # Compute mean values
                    mean_donor = np.mean(donor_selected)
                    mean_acceptor = np.mean(acceptor_selected)
                    mean_fret = calculate_fret(mean_donor, mean_acceptor)
                    mean_fret_list.append(mean_fret)

                    donor_fret_end = selected_regions[file_index][trace_index][1]
                    acceptor_fret_end = selected_regions[file_index][trace_index][3]
                    print("donor FRET end", donor_fret_end)
                    print("acceptor FRET end", acceptor_fret_end)
                    if donor_fret_end > acceptor_fret_end:
                        donor_outside_fret = donor_2[trace_index, FRET_end+1:donor_fret_end]
                        mean_donor_outside = np.mean(donor_outside_fret)
                        fret_outside_fret_alpha = fret_2[trace_index, FRET_end+1:donor_fret_end]
                        mean_fret_outside_alpha = np.mean(fret_outside_fret_alpha)
                        alpha_factor = mean_fret_outside_alpha/ mean_donor_outside
                        print("FRET END+1", FRET_end+1)
                        print("alpha:", alpha_factor)
                        alpha_factor_list.append(alpha_factor)

                    if acceptor_fret_end > donor_fret_end:
                        acceptor_outside_fret = acceptor_2[trace_index,FRET_end+1:acceptor_fret_end]
                        mean_acceptor_outside = np.mean(acceptor_outside_fret)
                        fret_outside_fret_beta = fret_2[trace_index, FRET_end+1:acceptor_fret_end]
                        mean_fret_outside_beta = np.mean(fret_outside_fret_beta)
                        beta_factor = mean_fret_outside_beta/ mean_acceptor_outside
                        print("FRET END+1", FRET_end+1)
                        print("beta:", beta_factor)
                        beta_factor_list.append(beta_factor)



                    # Plot histogram for FRET efficiency per frame
                    plt.figure(figsize=(8, 6))
                    plt.hist(fret_values, bins=20, color='green', edgecolor='black')
                    plt.xlim(0, 1)
                    plt.xlabel("FRET Efficiency")
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of FRET Efficiency for Trace {trace_index}, Frames {FRET_start}-{FRET_end}")
                    plt.savefig(os.path.join(save_folder, f"fret_histogram_trace_{trace_index}.png"))
                    plt.close()

                
        
        mean_alpha_factor = np.mean(alpha_factor_list) if alpha_factor_list else 1
        mean_beta_factor = np.mean(beta_factor_list) if beta_factor_list else 1

        print ("Alpha Factor:", mean_alpha_factor)
        print ("Beta Factro:", mean_beta_factor)

        # Save selected regions
        selected_df = pd.DataFrame.from_dict(selected_regions, orient='index', columns=['Start Frame', 'End Frame'])
        selected_df.index.name = 'Trace Index'
        selected_df.to_csv(os.path.join(save_folder, 'selected_regions.csv'))
    
        # Save FRET efficiency list
        fret_df = pd.DataFrame({'FRET Efficiency': fret_efficiency_list_apparent})
        fret_df.to_csv(os.path.join(save_folder, 'fret_efficiency_list.csv'), index=False)

        #Save mean FRET list
        mean_fret_df = pd.DataFrame({'Mean FRET Efficiency': mean_fret_list})
        mean_fret_df.to_csv(os.path.join(save_folder, 'mean_fret_efficiency_list.csv'))

        # Plot histogram of mean FRET efficiency values
        plt.figure(figsize=(8, 6))
        plt.hist(mean_fret_list, bins=20, color='blue', edgecolor='black')
        plt.xlim(0,1)
        plt.xlabel("apparent FRET Efficiency")
        plt.ylabel("Frequency")
        plt.title("Histogram of mean FRET Efficiency ")
        plt.savefig(os.path.join(save_folder, "mean_fret_histogram.png"))
        plt.show()

        # Plot histogram of mean FRET efficiency values
        plt.figure(figsize=(8, 6))
        plt.hist(fret_efficiency_list_apparent, bins=20, color='blue', edgecolor='black')
        plt.xlim(0,1)
        plt.xlabel("apparent FRET Efficiency")
        plt.ylabel("Frequency")
        plt.title("Total Histogram of FRET Efficiency")
        plt.savefig(os.path.join(save_folder, "total_apparent_fret_histogram.png"))
        plt.show()

        return mean_alpha_factor, mean_beta_factor
    

    @magicgui(
            call_button="Calculate Gamma factor",
            )
    # Function to plot corrected Fret
    def calculate_gamma_factor():
        global mean_gamma_factor
        if not selected_regions:
            print("No regions selected!")
            return
        '''if mean_alpha_factor is None or mean_beta_factor is None:
            mean_alpha_factor, mean_beta_factor = plot_apparent_fret()
        else:
            mean_alpha_factor,mean_beta_factor = mean_alpha_factor, mean_beta_factor'''
    
        gamma_factor_list = []
        
        for file_index, traces in selected_regions.items():
            donor_file_3 = donor_files[file_index]
            acceptor_file_3 = acceptor_files[file_index]
            fret_file_3 = FRET_files[file_index]

            donor_3 = donor_file_3.to_numpy()
            acceptor_3 = acceptor_file_3.to_numpy()
            fret_3 = fret_file_3.to_numpy()
            
                
            for trace_index, (donor_start, donor_end, acceptor_start, acceptor_end, FRET_start, FRET_end) in traces.items():    
                    print(donor_3.shape)
                    print("trace index", trace_index)

                    corrected_acceptor_intensity_before_list = []
                    corrected_acceptor_intensity_after_list = []

                    donor_fret_end = selected_regions[file_index][trace_index][1]
                    acceptor_fret_end = selected_regions[file_index][trace_index][3]
                    if donor_fret_end > acceptor_fret_end:

                        donor_selected_corect_before = donor_3[trace_index, donor_start:FRET_end]
                        acceptor_selected_correct_before = acceptor_3[trace_index, donor_start:FRET_end]
                        fret_selected_correct_before = fret_3[trace_index, donor_start:FRET_end]

                        donor_selected_corect_after = donor_3[trace_index, FRET_end+1:donor_end]
                        acceptor_selected_correct_after = acceptor_3[trace_index, FRET_end+1:donor_end]
                        fret_selected_correct_after = fret_3[trace_index, FRET_end+1:donor_end]


                        corrected_acceptor_intensity_values_before = [correct_acceptor(d, a, f, mean_alpha_factor, mean_beta_factor) for d, a, f in zip(donor_selected_corect_before, acceptor_selected_correct_before , fret_selected_correct_before)]
                        corrected_acceptor_intensity_before_list.append(corrected_acceptor_intensity_values_before)
                        mean_acceptor_before = np.mean(corrected_acceptor_intensity_before_list)
                        print("acceptor before", mean_acceptor_before)

                        corrected_acceptor_intensity_values_after = [correct_acceptor(d, a, f, mean_alpha_factor, mean_beta_factor) for d, a, f in zip(donor_selected_corect_after, acceptor_selected_correct_after , fret_selected_correct_after)]
                        corrected_acceptor_intensity_after_list.append(corrected_acceptor_intensity_values_after)
                        mean_acceptor_after = np.mean(corrected_acceptor_intensity_after_list)
                        print("acceptor after", mean_acceptor_after)


                        deltaI_DA = mean_acceptor_before - mean_acceptor_after
                        print("delta I_DA", deltaI_DA)

                        mean_donor_before = np.mean(donor_selected_corect_before)
                        mean_donor_after = np.mean(donor_selected_corect_after)

                        deltaI_DD = mean_donor_after-mean_donor_before
                        print("delta I_DD", deltaI_DD)

                        gamma_factor = deltaI_DA/deltaI_DD
                        print(gamma_factor)
                        gamma_factor_list.append(gamma_factor)

        mean_gamma_factor = np.mean(gamma_factor_list) if gamma_factor_list else 1
        print("gamma factor:", mean_gamma_factor)

        return mean_gamma_factor
    
    @magicgui(call_button="Plot corrected FRET")
    def plot_corrected_fret():
        if not selected_regions:
            print("No regions selected!")
            return
        '''if mean_alpha_factor is None or mean_beta_factor is None:
            mean_alpha_factor, mean_beta_factor = plot_apparent_fret()
        else:
            mean_alpha_factor,mean_beta_factor = mean_alpha_factor, mean_beta_factor
        
        if mean_gamma_factor is None:
            mean_gamma_factor = calculate_gamma_factor()
        else:
            mean_gamma_factor = mean_gamma_factor'''
        
    
        fret_efficiency_list_correted = []
        frame_wise_fret = []
        
        for file_index, traces in selected_regions.items():
            donor_file_4 = donor_files[file_index]
            acceptor_file_4 = acceptor_files[file_index]
            fret_file_4 = FRET_files[file_index]

            donor_4 = donor_file_4.to_numpy()
            acceptor_4 = acceptor_file_4.to_numpy()
            fret_4 = fret_file_4.to_numpy()
                
            for trace_index, (donor_start, donor_end, acceptor_start, acceptor_end, FRET_start, FRET_end) in traces.items():    
                    print(donor_4.shape)
                    print("trace index", trace_index)

                    donor_selected_corrected = donor_4[trace_index, FRET_start:FRET_end]
                    acceptor_selected_corrected = acceptor_4[trace_index, FRET_start:FRET_end]
                    fret_selected_corrected = fret_4[trace_index, FRET_start:FRET_end]

                    fret_values = [calculate_corrected_fret(d, a, f, mean_alpha_factor, mean_beta_factor, mean_gamma_factor) for d, a, f in zip(donor_selected_corrected, acceptor_selected_corrected, fret_selected_corrected)]
                    fret_efficiency_list_correted.extend(fret_values)
                    frame_wise_fret.append(fret_values)


                    # Plot histogram for FRET efficiency per frame
                    plt.figure(figsize=(8, 6))
                    plt.hist(fret_values, bins=20, color='green', edgecolor='black')
                    plt.xlim(0, 1)
                    plt.xlabel("corrected FRET Efficiency")
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of corrected FRET Efficiency for Trace {trace_index}, Frames {FRET_start}-{FRET_end}")
                    plt.savefig(os.path.join(save_folder, f"corrected fret_histogram_trace_{trace_index}.png"))
                    plt.close()

                    
        # Save FRET efficiency list
        fret_df = pd.DataFrame({'FRET Efficiency': fret_efficiency_list_correted})
        fret_df.to_csv(os.path.join(save_folder, 'fret_efficiency_list.csv'), index=False)

        # Plot histogram of mean FRET efficiency values
        plt.figure(figsize=(8, 6))
        plt.hist(fret_efficiency_list_correted, bins=20, color='blue', edgecolor='black')
        plt.xlim(0,1)
        plt.xlabel("corrected FRET Efficiency")
        plt.ylabel("Frequency")
        plt.title("Total Histogram of FRET Efficiency")
        plt.savefig(os.path.join(save_folder, "total_corrected_fret_histogram.png"))
        plt.show()




# Napari viewer
#viewer = napari.Viewer()
viewer.open(save_folder)
viewer.window.add_dock_widget(select_fret_region, area="right")
viewer.window.add_dock_widget(plot_apparent_fret, area="right")
viewer.window.add_dock_widget(calculate_gamma_factor, area="right")
viewer.window.add_dock_widget(plot_corrected_fret, area="right")
napari.run()