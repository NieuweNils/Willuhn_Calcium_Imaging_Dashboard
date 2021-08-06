import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def retrieve_metadata(data_after_cnmf_e):
    """
    :param data_after_cnmf_e: a matlab variable retrieved after using the CNMF_E algorithm on Ca-imaging data
    :return: an "metadata" dictionary with all retrieved "options" variables
    """
    names_dict = data_after_cnmf_e["options"][0][0].dtype
    names = [name for name in names_dict.fields.keys()]

    metadata = {}
    for i in range(26):
        option = data_after_cnmf_e["options"][0][0][0][0][i][0]
        if isinstance(option, str):
            metadata[names[i]] = option
        elif isinstance(option, np.ndarray):
            metadata[names[i]] = option[0]
    return metadata


def get_pixel_df(locations_df):
    number_of_neurons = locations_df.shape[1]
    non_zero_df = locations_df[locations_df != 0]
    pixels_containing_neurons_stack = non_zero_df.stack()

    # Select rows (pixels) that contain the column (neuron) of interest
    # TODO: Factorise this?
    pixels_all_neurons = []
    for i in range(number_of_neurons):
        pixels = [pixel for (pixel, neuron) in pixels_containing_neurons_stack.keys() if neuron == i]
        pixels_all_neurons.append(pixels)

    # Put data in a dataframe
    # NB: this pads the shorter lists with NaNs
    # TODO: Factorise this? (use numpy in 3d instead of pandas Dataframe)
    pixel_df = pd.DataFrame(pixels_all_neurons)
    return pixel_df


def get_col_and_row_df(pixel_df, metadata):
    d1 = metadata["d1"]  # width of recording in #pixels (or was it height?)
    col_df = pixel_df.apply(lambda x: x // d1)
    row_df = pixel_df.apply(lambda x: x % d1)
    return col_df, row_df


def get_cols_and_rows(pixel_df, metadata):
    col_df, row_df = get_col_and_row_df(pixel_df, metadata)
    cols = np.array(col_df)
    rows = np.array(row_df)
    return cols, rows


def get_mean_locations(locations_df, metadata):
    number_of_neurons = locations_df.shape[1]
    pixel_df = get_pixel_df(locations_df)

    # use dimensions d1 & d2 to figure out where the pixels are
    # TODO find out how to unpack this vector into (row,col) (even though there's padding)
    col_df, row_df = get_col_and_row_df(pixel_df, metadata)

    # Calculate the mean values for the dataframe
    col_df_mean = col_df.mean(axis=1)
    row_df_mean = row_df.mean(axis=1)

    mean_locations = []
    for i in range(number_of_neurons):
        mean_locations.append((col_df_mean[i], row_df_mean[i]))
    mean_locations_df = pd.DataFrame(mean_locations)

    return mean_locations_df


def distances(mean_locations_df):
    # Calculate pairwise euclidian distances
    distance_df = pd.DataFrame(itertools.combinations(mean_locations_df.index, 2), columns=["neuron_1", "neuron_2"])
    distance_df["distance"] = pdist(mean_locations_df.values, "euclid")

    return distance_df


def create_neighbour_dict(distance_correlation_df):
    neighbour_dict = {}
    new_row = 0

    # fill the dictionary as such: {key: value} -> {cell_number: row_number}
    for (neuron1, neuron2, *_) in distance_correlation_df.values:
        if neuron1 not in neighbour_dict:
            if neuron2 not in neighbour_dict:
                # neither neurons are associated with a previous row, assign to new row
                neighbour_dict[neuron1] = new_row
                neighbour_dict[neuron2] = new_row
                new_row += 1
            else:
                # neuron 2 is already in there, put neuron 1 in the same row
                old_row_2 = neighbour_dict[neuron2]
                neighbour_dict[neuron1] = old_row_2
        else:
            # neuron 1 is already in there
            if neuron2 not in neighbour_dict:
                # neuron 2 is not, assign to row of neuron 1
                old_row_1 = neighbour_dict[neuron1]
                neighbour_dict[neuron2] = old_row_1
            # if none of the above is True: both are already in separate rows, nothing to see here.

    return neighbour_dict


def a_neurons_neighbours(distance_df, correlation_df, max_distance=10, min_correlation=0.0):
    correlation_series = correlation_df.stack()

    df = distance_df
    df['correlation'] = correlation_series.values
    df = df[df['distance'] < max_distance]
    df = df[df['correlation'] > min_correlation]

    neighbour_dict = create_neighbour_dict(df)
    rows = set(neighbour_dict.values())
    # Create the table as a list of lists
    neighbours = []
    for row in rows:
        cells = [cell_number for cell_number, row_number in neighbour_dict.items() if row_number == row]
        neighbours.append(cells)
    # return a Dataframe with columns
    neighbour_df = pd.DataFrame(neighbours)
    for column in neighbour_df.columns:
        if column == 0:
            neighbour_df.rename(columns={column: "neuron"}, inplace=True)
        else:
            neighbour_df.rename(columns={column: f"neighbour_{column}"}, inplace=True)

    return neighbour_df


def correlating_neurons(fluorescence_traces):
    correlation_matrix = np.corrcoef(fluorescence_traces)
    correlation_matrix = np.absolute(correlation_matrix)

    correlation_df = pd.DataFrame(correlation_matrix)
    # Discard the lower left triangle as all correlation values will end up as doubles (includes the diagonal of 1.0's)
    correlation_df = correlation_df.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

    return correlation_df


def delete_locations(df, delete_list):
    print("deleting cells from locations_df")
    df = df.drop(delete_list, axis=1)  # pixel locations are stored column based (for a reason I don't remember)
    return df


def delete_traces(trace_dict, delete_list):
    print("deleting cells from traces")
    for cell in delete_list:
        trace_dict.pop(str(cell), None)
    return trace_dict


def delete_neighbours(df, delete_list):
    print("deleting cells from neighbours_df")
    # take out the cells in the delete_list
    df = df[~df.isin(delete_list)]
    # drop all rows that are completely empty
    df = df.dropna(how="all")
    # shift all the values to the left that were next to NaN values
    df = df.apply(lambda row: shift_away_nans(row), axis=1)
    # drop all cols that are completely empty
    df = df.dropna(how="all", axis=1)
    return df


def delete_neurons_distances(df, delete_list):
    print("deleting cells from distance_df")
    df = df[~df["neuron_1"].isin(delete_list)]
    df = df[~df["neuron_2"].isin(delete_list)]

    return df


def shift_away_nans(row):
    passed_nan = False
    passed_value = False
    for i in range(len(row)-1, -1, -1):  # NB: looping from the end of the array to the beginning (stop after 0 (# -1))
        if row[i] is None or np.isnan(row[i]):  # index is NaN, might shift this # NB: stupid dash changes stuff to None
            if passed_value:
                row = row[:i].append(row[i:].shift(-1, fill_value=np.nan))
            if not passed_nan:
                passed_nan = True
        else:  # index contains a value, not shifting this
            if not passed_value:
                passed_value = True
    return row


def merge_locations(locations, merge_list):
    print("merging locations")
    # take out the locations of the cells that you are merging
    updated_locations = locations.drop(merge_list[1:], axis=1)  # pixel locations are column based (don't remember why)
    # update first cell in list with updated locations
    merged_locations = locations[merge_list].max(axis=1)  # make a pixel 2 that is 2 in at least 1 cell
    updated_locations[merge_list[0]] = merged_locations
    return updated_locations


def merge_traces(traces, merge_list):
    print("merging cell traces")
    # take out the traces that you are merging
    trace_list = [traces[str(cell)] for cell in merge_list]
    for cell in merge_list[1:]:   # keep the first one (that one will store the merge)
        traces.pop(str(cell), None)
    # updated_traces = np.delete(traces, merge_list[1:], axis=0)

    # calculate the average trace to replace the ones you just deleted
    average_trace = np.mean(np.array(trace_list), axis=0)
    # add average trace to the traces (use the first of the list to store it)
    traces[str(merge_list[0])] = average_trace
    return traces
