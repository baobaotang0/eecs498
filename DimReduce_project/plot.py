import os
import numpy as np
import matplotlib.pyplot as plt

def get_model_name_from_filename(filename):
    for idx, letter in enumerate(filename):
        if letter == "_":
            return filename[:idx]


if __name__ == '__main__':
    Isomap_acc = np.array([[2, 0.5173749923706055, 0.24660000205039978], [10, 0.6902291774749756, 0.42640000581741333],
                  [20, 0.7135833501815796, 0.4959000051021576], [30, 0.7205625176429749, 0.45329999923706055],
                  [40, 0.7294999957084656, 0.4487999975681305], [50, 0.7342291474342346, 0.41190001368522644],
                  [60, 0.7383333444595337, 0.38040000200271606]])
    Isomap_t = np.array([[2, 78.0998923778534], [10, 156.64033818244934], [20, 154.20766711235046], [30, 159.63111972808838],
                [40, 161.25903701782227], [50, 164.73695969581604], [60, 214.7740249633789]])

    LocallyLinearEmbedding_acc = np.array([[2, 0.44091665744781494, 0.3043999969959259],
                                  [10, 0.503125011920929, 0.17399999499320984],
                                  [20, 0.5740416646003723, 0.24369999766349792],
                                  [30, 0.5922291874885559, 0.29679998755455017],
                                  [40, 0.5678125023841858, 0.19609999656677246],
                                  [50, 0.5896666646003723, 0.2160000056028366],
                                  [60, 0.5386041402816772, 0.21459999680519104]])
    LocallyLinearEmbedding_t = np.array([[2, 31.4595046043396], [10, 79.28730416297913], [20, 77.97422790527344],
                                [30, 78.73025012016296],
                                [40, 78.69000768661499], [50, 80.7215404510498], [60, 92.49149656295776]])

    SpectralEmbedding_acc = np.array([[2, 0.3097708225250244, 0.1657000035047531], [10, 0.4949375092983246, 0.3449000120162964],
                             [20, 0.4858333468437195, 0.4043999910354614], [30, 0.3017083406448364, 0.275299996137619],
                             [40, 0.36281248927116394, 0.24230000376701355],
                             [50, 0.36029165983200073, 0.23680000007152557],
                             [60, 0.4749166667461395, 0.28290000557899475]])
    SpectralEmbedding_t = np.array([[2, 131.1797707080841], [10, 135.31001472473145], [20, 135.71342492103577],
                           [30, 146.96659088134766],
                           [40, 153.12185764312744], [50, 161.3405840396881], [60, 219.7109980583191]])

    ax_tr_t, ax_te_t, ax_tr_acc, ax_te_acc = plt.figure().add_subplot(),plt.figure().add_subplot(), \
                                             plt.figure().add_subplot(),plt.figure().add_subplot()
    ax_tr_t.set_title("duration of fitting")
    ax_te_t.set_title("duration of predicting")
    ax_tr_acc.set_title("accuracy of training")
    ax_te_acc.set_title("accuracy of testing")
    ax_tr_acc.set_xlabel("reduced dimension")
    ax_te_acc.set_xlabel("reduced dimension")
    ax_tr_t.set_xlabel("reduced dimension")
    ax_te_t.set_xlabel("reduced dimension")
    for filename in os.listdir(os.getcwd()):
             if filename.endswith("npy"):
                print(filename)
                model_name = get_model_name_from_filename(filename)
                if "_acc" in filename:
                    with open(filename, 'rb') as f:
                        a = np.load(f)
                    ax_tr_acc.plot(a[:, 0], a[:, 1], label=model_name)
                    ax_te_acc.plot(a[:, 0], a[:, 2], label=model_name)
                if "_t" in filename:
                    with open(filename, 'rb') as f:
                        a = np.load(f)
                    ax_tr_t.plot(a[:, 0], a[:, 1], label=model_name)
                    ax_te_t.plot(a[:, 0], a[:, 2], label=model_name)

    ax_tr_acc.plot(Isomap_acc[:, 0], Isomap_acc[:, 1], label="Isomap")
    ax_te_acc.plot(Isomap_acc[:, 0], Isomap_acc[:, 2], label="Isomap")

    ax_tr_acc.plot(LocallyLinearEmbedding_acc[:, 0], LocallyLinearEmbedding_acc[:, 1], label="LocallyLinearEmbedding")
    ax_te_acc.plot(LocallyLinearEmbedding_acc[:, 0], LocallyLinearEmbedding_acc[:, 2], label="LocallyLinearEmbedding")
    ax_tr_acc.plot(SpectralEmbedding_acc[:, 0], SpectralEmbedding_acc[:, 1], label="SpectralEmbedding")
    ax_te_acc.plot(SpectralEmbedding_acc[:, 0], SpectralEmbedding_acc[:, 2], label="SpectralEmbedding")


    ax_te_t.plot(Isomap_t[:, 0], Isomap_t[:, 1], label="Isomap")
    ax_te_t.plot(LocallyLinearEmbedding_t[:, 0], LocallyLinearEmbedding_t[:, 1], label="LocallyLinearEmbedding")
    ax_te_t.plot(SpectralEmbedding_t[:, 0], SpectralEmbedding_t[:, 1], label="SpectralEmbedding")

    ax_tr_acc.legend()
    ax_te_acc.legend()
    ax_tr_t.legend()
    ax_te_t.legend()

    plt.show()





