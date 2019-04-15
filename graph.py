import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', default="/data/rl_rnd", help='The base data/output directory')
    parser.add_argument('--game', default='PixelCopter-v0', help='Game to use')
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--log_filename', default='log.txt', help='')
    parser.add_argument('--average_over', type=int, default=1, help='To smooth the graphed curves a bit')
    parser.add_argument('--pause_between_updates', type=int, default=10, help='In seconds')
    args = parser.parse_args()

    plt.ion()

    log_file = os.path.join(args.data_dir, args.game, args.experiment_name, args.log_filename)

    while True:
        with open(log_file) as f:
            content = f.readlines()
        means = []
        stds = []
        mins = []
        maxs = []
        for line in content:
            line = line.rstrip()
            if "avg cum rwd" in line:
                fields = line.split(":")
                mean = float(fields[4].split(",")[0])
                std = float(fields[5].split(",")[0])
                min = float(fields[6].split(",")[0])
                max = float(fields[7])
                means.append(mean)
                stds.append(std)
                mins.append(min)
                maxs.append(max)

        x = np.arange(len(means))
        means = np.array(means)
        stds = np.array(stds)

        divide = args.average_over
        means_means = []
        x_means = []
        stds_means = []
        for i in range(means.shape[0] // divide):
            x_means.append(x[i * divide])
            means_means.append(means[i * divide:(i + 1) * divide].mean())
            stds_means.append(stds[i * divide:(i + 1) * divide].mean())
        means_means = np.array(means_means)
        x_means = np.array(x_means)
        stds_means = np.array(stds_means)

        plt.plot(x_means, means_means)
        # plt.scatter(x, mins)
        # plt.scatter(x, maxs)
        plt.fill_between(x_means, means_means - stds_means, means_means + stds_means, alpha=0.5)

        plt.draw()
        plt.pause(args.pause_between_updates)
        plt.clf()


if __name__ == '__main__':
    main()
