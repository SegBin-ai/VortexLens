import matplotlib.pyplot as plt
import time

def plot_matches_count(file_path= "C:\\Users\\Aaditya Voruganti\\Desktop\\VortexLens\\screw_exploration\\Blob_detection\\matches_count.txt"):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of Good Matches')
    ax.set_title('Good Matches per Frame')
    line, = ax.plot([], [], 'r-')

    x_data = []
    y_data = []

    while True:
        with open(file_path, "r") as f:
            lines = f.readlines()
            y_data = [int(line.strip()) for line in lines]
            x_data = list(range(len(y_data)))
        
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.5)

if __name__ == '__main__':
    plot_matches_count()
