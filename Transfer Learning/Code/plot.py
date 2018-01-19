import matplotlib.pyplot as plt
import numpy as np

def plot(RESULT_FILE):
    xvalues = range(50, 650 + 1, 50)
    functions = ['2 Classes', '3 Classes', '4 Classes', '5 Classes', '6 Classes', '7 Classes', '8 Classes',
                 '9 Classes', '10 Classes', '11 Classes', '12 Classes', '13 Classes', '14 Classes', '15 Classes']
    data = np.loadtxt(RESULT_FILE, dtype=float).reshape(14,13)

    xmin = 50
    xmax = 650
    xstep = 50

    ymin = 980
    ymax = 1000
    ystep = 4

    colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(ymin/10, ymax/10)
    plt.xlim(xmin, xmax)

    plt.yticks([y / 10.0 for y in range(ymin, ymax + 1, ystep)], [str(x) + '%' for x in [y / 10.0 for y in range(ymin, ymax + 1, ystep)]], fontsize=14)
    #plt.yticks(range(ymin, ymax + 1, ystep), [str(x) + '%' for x in range(ymin, ymax + 1, ystep)], fontsize=14)
    plt.xticks(range(xmin, xmax + 1, xstep), [str(x) for x in range(xmin, xmax + 1, xstep)], fontsize=14)

    for y in [y / 10.0 for y in range(ymin, ymax + 1, ystep)]:
    #for y in range(ymin, ymax + 1, ystep):
        plt.plot(range(xmin, xmax + 1), [y] * len(range(xmin, xmax + 1)), "--", lw=0.5, color="black", alpha=0.3)

    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    for rank, column in enumerate(functions):
        plt.plot(xvalues, data[rank], lw=2.5, color=colors[rank])
        y_pos = data[rank][len(xvalues) - 1:]
        if column == '2 Classes':
            y_pos += 0.05
        elif column == '3 Classes':
            y_pos -= 0.0
        elif column == '4 Classes':
            y_pos -= 0.05
        elif column == '5 Classes':
            y_pos -= 0.04
        elif column == '6 Classes':
            y_pos -= 0.03
        elif column == '7 Classes':
            y_pos -= 0.02
        elif column == '8 Classes':
            y_pos += 0.02
        elif column == '9 Classes':
            y_pos -= 0.03
        elif column == '10 Classes':
            y_pos -= 0.02
        elif column == '11 Classes':
            y_pos += 0.02
        elif column == '12 Classes':
            y_pos -= 0.02
        elif column == '13 Classes':
            y_pos -= 0.04
        elif column == '14 Classes':
            y_pos -= 0.01
        elif column == '15 Classes':
            y_pos -= 0.06
        plt.text(xmax * 1.01, y_pos, column, fontsize=14, color=colors[rank])
    plt.title('Analysis of Transfer Learning on Animals with Attributes 2 dataset', fontsize=18)
    plt.xlabel('Number of images', fontsize = 14)
    plt.ylabel('Accuracy', fontsize = 14)
    plt.show()