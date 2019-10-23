import numpy
from scipy.optimize import curve_fit


def normal_distribution(x, a, mean, std):
    return a*numpy.exp(-(x-mean)**2/(2*std**2))


def line(x, a, b):
    return a*x + b


def do_gaussian_fit(data_x_axis, data_y_axis, noise_points=3):
    noise = get_noise_function(data_x_axis, data_y_axis, noise_points)
    data_y_axis = [y-noise(x) for x, y in zip(data_x_axis, data_y_axis)]
    y_max = numpy.max(data_y_axis)
    mean = numpy.mean(data_x_axis)
    std = numpy.std(data_x_axis)
    optimized_parameters, parameters_covariance = \
        curve_fit(normal_distribution, data_x_axis, data_y_axis,
                  p0=[y_max, mean, std])
    optimized_gaussian = lambda x: normal_distribution(x, *optimized_parameters) + noise(x) # noqa
    return optimized_gaussian, optimized_parameters, parameters_covariance


def get_noise_function(x_data, y_data, point_number):
    x_data = list(x_data)
    y_data = list(y_data)

    regression_data = [x_data[:point_number]+x_data[-point_number:],
                       y_data[:point_number]+y_data[-point_number:]]

    optimized_parameters, parameters_covariance = \
        curve_fit(line, *regression_data)
    noise = lambda x: line(x, *optimized_parameters) # noqa
    return noise
