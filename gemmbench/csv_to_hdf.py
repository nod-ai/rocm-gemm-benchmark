import pandas as pd
import h5py
import numpy as np
import sys

def csv_to_hdf5(csv_file: str, hdf5_file: str):
    """
    Convert a CSV file to an HDF5 file.
    
    :param csv_file: Path to the input CSV file
    :param hdf5_file: Path to the output HDF5 file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create an HDF5 file
    with h5py.File(hdf5_file, 'w') as hf:
        # Create a group for the dataset
        group = hf.create_group('data')
        
        # Store each column as a dataset
        for column in df.columns:
            # Convert to appropriate numpy dtype
            if df[column].dtype == 'object':
                # For string columns, use variable-length string dtype
                dt = h5py.special_dtype(vlen=str)
                data = df[column].astype(str).to_numpy()
            else:
                # For numeric columns, use the corresponding numpy dtype
                dt = df[column].dtype
                data = df[column].to_numpy()
            
            # Create dataset
            group.create_dataset(column, data=data, dtype=dt)
        
        # Store column names as an attribute
        group.attrs['columns'] = df.columns.tolist()

    print(f"CSV file '{csv_file}' has been converted to HDF5 file '{hdf5_file}'")

if __name__ == '__main__':
    csv_to_hdf5(sys.argv[1], sys.argv[2])
