import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist

import scipy.sparse as sparse
from past.utils import old_div


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


def distances(mean_locations_dict):
    # extract the data from the dictionary
    mean_locations = np.array([array for array in mean_locations_dict.values()])
    # make a table of all combinations of neurons
    neuron_pairs = np.array(list(itertools.combinations(mean_locations_dict.keys(), 2)))
    # Calculate pairwise euclidian distances
    pairwise_distance = np.reshape(pdist(mean_locations, "euclid"), [-1, 1])
    # store the distance with the right combination of cell ids in a 2D np array
    distance_array = np.append(neuron_pairs,
                               pairwise_distance,
                               axis=1)

    return distance_array


def create_neighbour_dict(distance_correlation_table):
    neighbour_dict = {}
    new_row = 0

    # fill the dictionary as such: {key: value} -> {cell_number: row_number}
    for (neuron1, neuron2, *_) in distance_correlation_table:
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


def a_neurons_neighbours(distance_array, correlation_df, max_distance=10, min_correlation=0.0):
    # create table with distances and correlations
    correlation_series = correlation_df.stack()
    corr_array = np.reshape(np.array(correlation_series), [-1, 1])
    dist_array = distance_array
    dist_corr_array = np.append(dist_array, corr_array, axis=1)
    # filter to keep cells close enough to each other and with high enough correlation
    close_array = dist_corr_array[dist_corr_array[:, 2] < max_distance]
    neighbour_array = close_array[close_array[:, 3] > min_correlation]

    neighbour_dict = create_neighbour_dict(neighbour_array)
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
    cell_numbers = fluorescence_traces.keys()
    trace_values = np.array([array for array in fluorescence_traces.values()])
    correlation_matrix = np.corrcoef(trace_values)
    correlation_matrix = np.absolute(correlation_matrix)

    correlation_df = pd.DataFrame(correlation_matrix, columns=cell_numbers, index=cell_numbers)
    # Discard the lower left triangle as all correlation values will end up as doubles (includes the diagonal of 1.0's)
    correlation_df = correlation_df.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

    return correlation_df


def delete_locations_dict(loc_dict, delete_list):
    print("deleting cells from loc_dict")
    for cell in delete_list:
        loc_dict.pop(str(cell), None)
    return loc_dict


def delete_traces(trace_dict, delete_list):
    print("deleting cells from traces")
    for cell in delete_list:
        trace_dict.pop(str(cell), None)
    return trace_dict


def merge_locations_dict(locations, merge_list):
    print("merging locations (loc_dict)")
    # take out the locations of the cells that you are merging
    loc_list = [locations[str(cell)] for cell in merge_list]
    for cell in merge_list[1:]:  # keep the first one (that one will store the merge)
        locations.pop(str(cell), None)
    # take the highest energy in a given pixel to replace the ones you just deleted
    average_location = np.max(np.array(loc_list), axis=0)
    locations[str(merge_list[0])] = average_location

    # TODO: also update mean_locations here (and let that trigger update_neighbour_data)
    return locations


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


def get_centre_of_mass(loc_dict, d1, d2):
    """Calculation of the center of mass for spatial components

       From Caiman: https://github.com/flatironinstitute/CaImAn
       @author: agiovann

     Inputs:
     ------
     locations:     np.ndarray
          matrix of spatial components (pixels x number_of_cells)

     d1:            int
          number of pixels in x-direction

     d2:            int
          number of pixels in y-direction

     Output:
     -------
     center_of_mass:  np.ndarray
          center of mass for spatial components (number_of_cells x 2)
    """
    # extract the data from the dictionary
    locations = np.array([array for array in loc_dict.values()])
    nr_of_cells = np.shape(locations)[0]
    # initalise variables
    coordinates = {'x': np.kron(np.ones((d2, 1)),
                                np.expand_dims(list(range(d1)), axis=1)),
                   'y': np.kron(np.expand_dims(list(range(d2)), axis=1),
                                np.ones((d1, 1)))
                   }
    center_of_mass = np.zeros((nr_of_cells, 2))
    # calculate center of mass
    center_of_mass[:, 0] = old_div(np.dot(coordinates['x'].T, locations.T), locations.sum(axis=1))
    center_of_mass[:, 1] = old_div(np.dot(coordinates['y'].T, locations.T), locations.sum(axis=1))
    # put the data in a dictionary
    center_of_mass_dict = dict(zip(loc_dict.keys(), center_of_mass))
    return center_of_mass_dict


def retrieve_contour_coordinates(locations, background, thr=None, energy_threshold=0.9,
                      swap_dim=False,
                      **kwargs):
    """returns the coordinates of the contours of each cell in the locations tensor

       From Caiman: https://github.com/flatironinstitute/CaImAn
       @author: agiovann

     Parameters:
     -----------
     locations:     np.ndarray or sparse matrix
               Matrix of Spatial components (d x K)

     background:    np.ndarray (2D)
               Background image (e.g. mean, correlation)

     thr:           scalar between 0 and 1
               Energy threshold for computing contours (default 0.9)
               Kept for backwards compatibility.
               If not None then thr_method = 'nrg', and nrgthr = thr

     thr_method:    [optional] string
                Method of thresholding:
                'max' sets to zero pixels that have value less
                than a fraction of the max value
                'nrg' keeps the pixels that contribute up to a
                specified fraction of the energy

     maxthr:        [optional] scalar
                Threshold of max value (default 0.2)

     nrgthr:        [optional] scalar
                Threshold of energy (default 0.9)

     colors:        string
                Color of the contour colormap (default "orange")

     Returns:
     --------
     contour_coordinates:   list
             list of contour plot coordinates for each cell
    """
    if sparse.issparse(locations):
        locations = np.array(locations.todense())
    else:
        locations = np.array(locations)

    if swap_dim:
        background = background.T
        print('Swapping dimensions')

    d1, d2 = np.shape(background)
    nr_pixels, nr_cells = np.shape(locations)

    if thr is not None:
        energy_threshold = thr
        warn("The way to call utilities.plot_contours has changed.")

    x, y = np.mgrid[0:d1:1, 0:d2:1]

    contour_coordinates = []
    for i in range(nr_cells):
        # TODO: check if this is still necessary
        # remove a contour plot if it was already drawn.
        for collection in plt.gca().collections:
            collection.remove()

        index = np.argsort(locations[:, i], axis=None)[::-1]
        cum_energy = np.cumsum(locations[:, i].flatten()[index]**2)
        cum_energy /= cum_energy[-1]  # normalise location vector
        location_vector = np.zeros(nr_pixels)
        location_vector[index] = cum_energy
        thr = energy_threshold

        if swap_dim:
            location_matrix = np.reshape(location_vector, np.shape(background), order='C')
        else:
            location_matrix = np.reshape(location_vector, np.shape(background), order='F')

        cell_contour = plt.contour(y, x, location_matrix, [thr])

        paths = cell_contour.collections[0].get_paths()
        coordinate_vector = np.atleast_2d([np.nan, np.nan])
        for path in paths:
            vertex = path.vertices
            nr_close_coordinates = np.sum(np.isclose(vertex[0, :], vertex[-1, :]))
            if nr_close_coordinates < 2:
                if nr_close_coordinates == 0:
                    new_point = np.round(old_div(vertex[-1, :], [d2, d1])) * [d2, d1]
                    vertex = np.concatenate((vertex, new_point[np.newaxis, :]), axis=0)
                else:
                    vertex = np.concatenate((vertex, vertex[0, np.newaxis]), axis=0)
            coordinate_vector = np.concatenate((coordinate_vector, vertex, np.atleast_2d([np.nan, np.nan])), axis=0)
        contour_coordinates.append(coordinate_vector)

    return contour_coordinates
