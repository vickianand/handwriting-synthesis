import numpy as np
from matplotlib import pyplot


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: " + save_name)

    pyplot.close()

# ======================== normalize the strokes offset ========================
def normalize_data1(strokes):
    '''
    normalize to [-1, 1] separately for x1 and x2
    '''
    strokes_cat = np.concatenate(list(strokes))

    (min1, max1) = strokes_cat[:, 1].min(), strokes_cat[:, 1].max()
    (min2, max2) = strokes_cat[:, 2].min(), strokes_cat[:, 2].max()
    range1 = max1 - min1
    range2 = max2 - min2
    for i in range(strokes.shape[0]):
        strokes[i][:, 1] = 2*(strokes[i][:, 1] - (min1 + range1/2.)) / range1
        strokes[i][:, 2] = 2*(strokes[i][:, 2] - (min2 + range2/2.)) / range2

    return strokes


def normalize_data2(strokes):
    '''
    normalize to [-1, 1] range using combined min and max of x1 and x2
    '''
    strokes_cat = np.concatenate(list(strokes))

    (min1, max1) = strokes_cat[:, 1:].min(), strokes_cat[:, 1:].max()
    # print("min1 = {}, max1 = {}".format(min1, max1))
    range1 = max1 - min1
    for i in range(strokes.shape[0]):
        strokes[i][:, 1:] = 2*(strokes[i][:, 1:] - (min1 + range1/2.)) / range1

    return strokes


def normalize_data3(strokes):
    '''
    normalize to mean 0 and standard deviation 1
    '''
    strokes_cat = np.concatenate(list(strokes))

    (mean, std) = strokes_cat[:, 1:].mean(), strokes_cat[:, 1:].std()
    
    # (mean1, std1) = strokes_cat[:, 1].mean(), strokes_cat[:, 1].std()
    # (mean2, std2) = strokes_cat[:, 2].mean(), strokes_cat[:, 2].std()
    
    for i in range(strokes.shape[0]):
        # strokes[i][:, 1:] = (strokes[i][:, 1:] - mean) / std
        strokes[i][:, 1:] = strokes[i][:, 1:] / std
        
        # strokes[i][:, 1] = (strokes[i][:, 1] - mean1) / std1
        # strokes[i][:, 2] = (strokes[i][:, 2] - mean2) / std2

    return strokes

