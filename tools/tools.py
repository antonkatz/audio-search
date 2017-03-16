import scipy as sp
import scipy.io.wavfile
import scipy.signal as ss
import numpy as np
from matplotlib import pyplot as plt


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
def sample_opt_constraint(data, num_samples, limitd, distance_const=np.nan, angle_constraint=np.nan):
    limitx = data.shape[1] - 1
    limity = data.shape[0] - 1

    samples = np.empty(num_samples)

    for i in range(0, num_samples):
        # if a constraint not present, set it to random
        a = angle_constraint
        d = distance_const
        if np.isnan(a):
            a = np.random.rand() * np.pi
        if np.isnan(d):
            d = np.random.rand() * limitd

        # setting up for the touch down point
        th_x = np.rint(d * np.cos(a))  # theta
        th_y = np.rint(d * np.sin(a))

        lx_low = zero_or_below(th_x)  # only an issue if left directed
        lx_high = limitx - zero_or_above(th_x)
        ly_low = zero_or_below(th_y)
        ly_high = limity - zero_or_above(th_y)  # only in issue if up directed

        x1 = np.around(np.random.randint(lx_low, lx_high))
        y1 = np.around(np.random.randint(ly_low, ly_high))

        # second point
        #
        # print("a,d", a * (180 / np.pi), d)
        # print("theta y,x", th_y, th_x)

        x2 = x1 + th_x
        y2 = y1 + th_y

        samples[i] = data[int(y1), int(x1)] - data[int(y2), int(x2)]
    return samples


def zero_or_above(x):
    if x < 0:
        return 0
    return x


def zero_or_below(x):
    if x < 0:
        return np.abs(x)
    return 0


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

    samples = np.empty((const_count, num_samples))
    for i in range(const_count):
        ac = angle_constraint[i]
        dc = distance_const[i]
        samples[i] = sample_opt_constraint(data=data, num_samples=num_samples, limitd=limitd,
                                           distance_const=dc, angle_constraint=ac)
    return samples
