import scipy as sp
import scipy.io.wavfile
import scipy.signal as ss
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import LineString


def todB(array):
    mean = np.mean(array)
    return 10 * np.log10(array / mean)


def plotSpectrogram(to_plot, time_points, yticks=np.array([]), cmap=plt.cm.gist_heat, subplot=111, _fig=False,
                    fig_size=(15, 7)):
    if (_fig == False):
        fig = plt.figure(figsize=fig_size)
    else:
        fig = _fig
    ax = fig.add_subplot(subplot)
    plt.imshow(to_plot, cmap=cmap)
    ax.set_aspect('auto')

    ## making pretty
    ax.set_ylim(ax.get_ylim()[::-1])  ## reversing axis
    plt.title("Spectrogram in the human hearing range")
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    num_labels = 10
    xpoints = np.arange(to_plot.shape[1], step=int(to_plot.shape[1] / num_labels))
    plt.xticks(xpoints, np.round(time_points[xpoints], 1))
    if (yticks.size > 0):
        ypoints = np.arange(to_plot.shape[0], step=int(to_plot.shape[0] / num_labels))
        plt.yticks(ypoints, yticks[ypoints])
    plt.colorbar(orientation='vertical')
    return fig


""":returns numpy array of [x,y]"""


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return np.array([x, y])


vpol2cart = np.vectorize(pol2cart)


def cartesian_coord(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points


def generateSpectrogram(audio_data, segment_length, sample_rate, overlap, cutoff_freq, cutoff_time):
    seg_lenght_time = segment_length / sample_rate
    # how much data to process each loop
    block_size = 10 * sample_rate

    data_size = sample_rate * cutoff_time
    if data_size > len(audio_data):
        data_size = len(audio_data)
    num_blocks = int(np.floor(data_size / block_size))

    spec_buffer = []
    time_points_buffer = []
    freq_points = None
    last_time_point = 0
    for bi in range(0, num_blocks):
        data_start = bi * block_size - overlap
        data_end = (bi + 1) * block_size
        if data_start < 0:
            data_start = 0
        data = audio_data[data_start: data_end]

        freq_points, t_points, spectrogram = ss.spectrogram(data, nperseg=segment_length, fs=sample_rate,
                                                            noverlap=overlap, mode="psd", nfft=sample_rate)
        cut_spectrogram = spectrogram[:cutoff_freq, ]
        spec_buffer.append(cut_spectrogram)

        t_points += last_time_point
        time_points_buffer.append(t_points)
        last_time_point = t_points[-1]

    full_spectrogram = np.hstack(spec_buffer)
    full_time = np.hstack(time_points_buffer)
    return freq_points, full_time, full_spectrogram


vfrand = np.vectorize(np.random.randint, otypes=[np.ndarray], excluded=["size"], signature="(),()->(m)")

# samples with optional constraints, replacing them with random values each individual sample point
# limitd is limit to distance
""":returns samples (2 columns: mean, sd)
            function_results a list of callback"""


def sample_opt_constraint(data, num_samples, limitd=None, distance_const=np.nan, angle_constraint=np.nan, callback=None):
    limitx = data.shape[1] - 1
    limity = data.shape[0] - 1
    if (limitd is None) & (np.isnan(distance_const) | np.isnan(angle_constraint)):
        raise Exception("limitd not set")
    # mean, sd
    samples = np.empty((num_samples, 2))
    for i in range(0, num_samples):
        # if a constraint not present, set it to random
        a = angle_constraint
        d = distance_const
        if np.isnan(a):
            a = np.random.rand() * np.pi
        if np.isnan(d):
            d = int(np.around(np.random.rand() * limitd, decimals=0))
            d = 1 if (d < 1) else d

        # setting up for the touch down point
        th_x = np.rint(d * np.cos(a))  # theta
        th_y = np.rint(d * np.sin(a))
        # limits
        lx_low = zero_or_below(th_x)  # only an issue if left directed
        lx_high = limitx - zero_or_above(th_x)
        ly_low = zero_or_below(th_y)
        ly_high = limity - zero_or_above(th_y)  # only in issue if up directed
        # points
        x1 = int(np.around(np.random.randint(lx_low, lx_high)))
        y1 = int(np.around(np.random.randint(ly_low, ly_high)))
        x2 = int(x1 + th_x)
        y2 = int(y1 + th_y)


        # analysis
        y1, y2 = order_for_slice(y1, y2)
        x1, x2 = order_for_slice(x1, x2)
        line_samples = np.empty(d + 1)
        line = LineString([(x1, y1), (x2, y2)])
        data_slice = data[y1:y2 + 1, x1:x2 + 1]
        for j in range(d + 1):
            c = line.interpolate(j / d, normalized=True)
            line_samples[j] = data_slice[int(c.y - y1), int(c.x - x1)]

        sd, mean = np.std(line_samples), np.mean(line_samples)
        samples[i, 0] = mean
        samples[i, 1] = sd
    return samples


def zero_or_above(x):
    if x < 0:
        return 0
    return x


def zero_or_below(x):
    if x < 0:
        return np.abs(x)
    return 0

def order_for_slice(x1, x2):
    if (x1 > x2):
        return x2, x1
    return x1, x2

def sample_with_constraint(data, num_samples, limitd, distance_const=None, angle_constraint=None):
    if (angle_constraint is None) & (distance_const is None):
        raise Exception("one constraint must be specified")

    # how many constraints there are
    if angle_constraint is None:
        angle_constraint = np.full(distance_const.shape, np.nan)
    if distance_const is None:
        distance_const = np.full(angle_constraint.shape, np.nan)

    # the size of the two constraints arrays must be equal
    if (distance_const.shape != angle_constraint.shape):
        raise Exception("constraint arrays are not of the same size")
    const_count = angle_constraint.shape[0]

    #
    #  NOT GOING TO WORK PROPERLY
    #

    samples = np.empty((const_count, num_samples))
    for i in range(const_count):
        ac = angle_constraint[i]
        dc = distance_const[i]
        samples[i] = sample_opt_constraint(data=data, num_samples=num_samples, limitd=limitd,
                                           distance_const=dc, angle_constraint=ac)
    return samples
