import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def retrieve_metadata(data_after_cnmf_e):
    """
    :param data_after_cnmf_e: a matlab variable retrieved after using the CNMF_E algorithm on Ca-imaging data
    :return: an "metadata" dictionary with all retrieved "options" variables
    """
    names_dict = data_after_cnmf_e['options'][0][0].dtype
    names = [name for name in names_dict.fields.keys()]

    metadata = {}
    for i in range(26):
        option = data_after_cnmf_e['options'][0][0][0][0][i][0]
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
    d1 = metadata['d1']  # width of recording in #pixels (or was it height?)
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


def shortest_distances(mean_locations_df, small_distance=10):
    # Calculate pairwise euclidian distances
    distance_df = pd.DataFrame(itertools.combinations(mean_locations_df.index, 2), columns=['neuron_1', 'neuron_2'])
    distance_df['distance'] = pdist(mean_locations_df.values, 'euclid')

    # Select neurons that are very close together
    # TODO: write test cases to check that algorithms work as intended
    neurons_closest_together_df = distance_df[distance_df['distance'] < small_distance]
    return neurons_closest_together_df


def create_neighbour_dict(neurons_close_to_another_df):
    neighbour_dict = {}
    new_row = 0

    # fill the dictionary as such: {key: value} -> {cell_number: row_number}
    for (neuron1, neuron2, distance) in neurons_close_to_another_df.values:
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


def a_neurons_neighbours(neurons_close_to_another_df):
    neighbour_dict = create_neighbour_dict(neurons_close_to_another_df)
    rows = set(neighbour_dict.values())
    # Create the table as a list of lists
    size_based_neuron_clusters = []
    for row in rows:
        cells = [cell_number for cell_number, row_number in neighbour_dict.items() if row_number == row]
        size_based_neuron_clusters.append(cells)

    # Convert to a Dataframe with columns
    size_based_neuron_clusters_df = pd.DataFrame(size_based_neuron_clusters)

    for column in size_based_neuron_clusters_df.columns:
        if column == 0:
            size_based_neuron_clusters_df.rename(columns={column: 'neuron'}, inplace=True)
        else:
            size_based_neuron_clusters_df.rename(columns={column: f'neighbour_{column}'}, inplace=True)

    return size_based_neuron_clusters_df


def correlating_neurons(fluorescence_traces):
    correlation_matrix = np.corrcoef(fluorescence_traces)
    correlation_matrix = np.absolute(correlation_matrix)

    correlation_df = pd.DataFrame(correlation_matrix)
    # Discard the lower left triangle as all correlation values will end up as doubles (includes the diagonal of 1.0's)
    correlation_df = correlation_df.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

    highly_correlating_neurons = correlation_df[correlation_df > 0.6]

    return highly_correlating_neurons
