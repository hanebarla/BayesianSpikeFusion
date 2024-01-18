import os, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle

from dataset import dataset_factory
from model_ann import model_factory
from model_snn import create_model_snn
from lsm import LSM, gauss_LSM, quation_LSM, gauss_func, log_gauss_func
from aic import l_MAX, AIC, BIC
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, Matern, ConstantKernel
from scipy import interpolate
from scipy.special import beta as Beta
from tqdm import trange

parser = argparse.ArgumentParser()
parser.add_argument('model', default='vgg11', type=str)
parser.add_argument('dataset', default='cifar10', type=str)
parser.add_argument('alpha_mode', type=str)  # grid or emp
parser.add_argument('--gpu', type=str)
parser.add_argument('--ic_index', type=int, default=-1)
parser.add_argument('--sim_time', type=int, default=3000)
parser.add_argument('--plot_alpha', type=int, default=1)
args = parser.parse_args()

BatchSize = 500
plt.rc('font', family='serif')

def spike_simulation(ic_index, work_dir, dataset_dir, sim_time):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    print()
    print(" --- simulation start --- ")

    train_dataloader, test_dataloader, num_classes, input_shape = dataset_factory(args.dataset, dataset_dir, BatchSize, normalize=False, augument=False)

    ann_model = model_factory(model_kind=args.model, ic_index=ic_index, activation="relu", num_classes=num_classes, input_channel=input_shape[0])
    spiking_model = create_model_snn(model=ann_model, batch_size=BatchSize, input_shape=input_shape)
    spiking_model.load_state_dict(torch.load(os.path.join(work_dir, "snn.pth")))

    spiking_model.to(device)
    spiking_model.eval()

    with torch.no_grad():
        tot = 0

        for i, data in enumerate(train_dataloader):
            print("--- {} batch start ---".format(i))
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            spiking_model.reset()

            tot += labels.size(0)
            print(tot)

            count = {}
            for index in spiking_model.classifiers.keys():
                count[int(index)] = torch.zeros((sim_time, BatchSize, num_classes), device=device)

            for t in (range(sim_time)):
                outputs = spiking_model(images)

                for index, out in outputs.items():
                    count[index][t,:,:] = out

            np.savez_compressed(os.path.join(work_dir,"labels_{}.npz".format(i)), l=labels.cpu().detach().numpy().copy())
            for index, cnt in count.items():
                np.savez_compressed(os.path.join(work_dir,"{}_output_{}.npz".format(index, i)), o=cnt.cpu().detach().numpy().copy())

            if tot >= 10000:
                break


def plot_alpha_estimation(ic_index, work_dir, dataset_dir, sim_time, mode="grid"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    print()
    print(" --- start alpha estimation --- ")

    try_num = int(10000 // BatchSize) + min(1, 10000 % BatchSize)
    indexes = [ic_index, -1]

    alphas = np.zeros((sim_time))
    betas = np.zeros((sim_time, 101))
    
    if args.dataset == "cifar100":
        num_classes = 100
    else:
        num_classes = 10
    
    for j in range(try_num):
        labels = np.zeros((BatchSize))
        count = {}
        for index in indexes:
            count[index] = np.zeros((sim_time, BatchSize, num_classes))
            
        labels_path = os.path.join(work_dir,"labels_{}.npz".format(j))
        tmp_labels = np.load(labels_path)["l"]
        labels = tmp_labels[np.newaxis,:,np.newaxis]
        
        for index in indexes:
            cnt_save_path = os.path.join(work_dir,"{}_output_{}.npz".format(index, j))
            tmp_fire = np.load(cnt_save_path)["o"]
            # print(tmp_fire[:15,:2,4])
            tmp_sum_cnt = np.cumsum(tmp_fire, axis=0)
            # print(tmp_sum_cnt[:15,:2,4])
            count[index] = tmp_sum_cnt
            # raise ValueError

        # labels = torch.from_numpy(labels.astype(np.float32)).clone()

        time_batch = 50
        tmp_alphas = np.arange(0, 101, 1, dtype="float64")
        prior_alphas = np.tile(tmp_alphas, (time_batch, BatchSize, num_classes, 1))
        for t in trange(int(sim_time//time_batch)):
            if mode == "grid":
                N = np.arange(t*time_batch+1,(t+1)*time_batch+1, 1)
                N = N[:,np.newaxis,np.newaxis]
                M = count[-1][t*time_batch:(t+1)*time_batch,:,:]
                E = (count[ic_index][t*time_batch:(t+1)*time_batch,:,:] / N) + 1e-16  # p_prior
                N = N[:,:,:,np.newaxis]
                M = M[:,:,:,np.newaxis]
                E = E[:,:,:,np.newaxis]
                # grid
                best_acc = 0
                best_alpha = 1

                prior_betas = prior_alphas*(1/E-1)
                posterior_alphas = prior_alphas + M
                posterior_betas = prior_betas + (N - M)

                fr = posterior_alphas / (posterior_alphas + posterior_betas)
                pred = np.argmax(fr, axis=2)
                accs = np.sum(pred == labels, axis=1)
                best_alpha_index = np.argmax(accs, axis=-1)
                best_alpha = tmp_alphas[best_alpha_index]

                # for prior_alpha in range(1, 101, 1):
                #     prior_beta = prior_alpha*(1/E - 1)
                #     posterior_alpha = prior_alpha + M
                #     posterior_beta = prior_beta + (N - M)
                #     fr = posterior_alpha / (posterior_alpha + posterior_beta)
                #     fr = torch.from_numpy(fr.astype(np.float32)).clone()
                #     print(fr.size())
                #     pred = torch.max(fr, 1)[1]
                #     print(pred.shape)
                #     raise ValueError
                #     # print(pred, labels)
                #     acc = torch.sum(pred == labels) / (BatchSize)
                #     if best_acc < acc:
                #         best_acc = acc
                #         best_alpha = prior_alpha
            elif mode == "emp":
                N = np.arange(t*time_batch+1,(t+1)*time_batch+1, 1)
                N = N[:,np.newaxis,np.newaxis]
                M = count[-1][t*time_batch:(t+1)*time_batch,:,:]
                E = (count[ic_index][t*time_batch:(t+1)*time_batch,:,:] / N) + 1e-16  # p_prior
                # emperial
                N = N[:,:,:,np.newaxis]
                M = M[:,:,:,np.newaxis]
                E = E[:,:,:,np.newaxis]
                beta_1 = Beta(M+prior_alphas+1, N-M+prior_alphas*(1/E-1))
                beta_2 = Beta(prior_alphas+1, prior_alphas*(1/E-1))
                # print(beta_2[0,:,:]+1e-16)
                beta = beta_1/(beta_2+1e-16)
                # beta = beta_1/beta_2
                # print(tmp_betas[0].shape)
                # print(tmp_betas[0])
                best_alpha_index = np.argmax(beta, axis=-1)
                # print(prior_alphas.shape)
                # print(best_alpha_index.shape)
                # print(np.max(best_alpha_index))
                # print(alphas[best_alpha_index].shape)
                best_alpha = np.mean(tmp_alphas[best_alpha_index], axis=(1,2))
                betas[t*time_batch:(t+1)*time_batch,:] += np.mean(beta, axis=(1,2)) / try_num
                # print(best_alpha.shape)
            else:
                raise ValueError

            alphas[t*time_batch:(t+1)*time_batch] += best_alpha / try_num  # mean
            # print(np.mean(beta, axis=(1,2)).shape)

        print(betas[0].shape)

    xs = np.arange(sim_time)
    # ys = np.array(alphas).mean(axis=1)
    ys = alphas
    np.savez(os.path.join(work_dir, "alpha_output_{}.npz".format(mode)), alpha=ys, beta=betas)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.imshow(betas.T, cmap="jet")
    fig.savefig(os.path.join(work_dir, "time_alpha_betas_{}.svg".format(mode)), dpi=300)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c="r", s=5)
    fig.savefig(os.path.join(work_dir, "time_alpha_mode_{}.svg".format(mode)))
    
    
def plot_alpha(ic_index, work_dir, dataset_dir, sim_time, alpha_mode, mode="grid"):
    xs = np.arange(sim_time)
    # ys = np.array(alphas).mean(axis=1)
    ys = np.load(os.path.join(work_dir, "alpha_output_{}.npz".format(alpha_mode)))["alpha"]
    betas = np.load(os.path.join(work_dir, "alpha_output_{}.npz".format(alpha_mode)))["beta"]
    print(np.isnan(betas))
    print(ys.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c="r", s=5, alpha=0.5)
    fig.savefig(os.path.join(work_dir, "time_alpha.svg"))
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax1.imshow(betas.T[:,::-1], cmap="jet")
    ax1.imshow(betas.T, cmap="jet", aspect="auto", origin='lower')
    fig.savefig(os.path.join(work_dir, "time_alpha_betas.svg"), dpi=300)
    
    
def get_approximate_curve(ic_index, work_dir, dataset_dir, sim_time, alpha_mode=None, mode="AIC"):
    xs = np.arange(sim_time)
    xs_unit = xs / sim_time
    ys = np.load(os.path.join(work_dir, "alpha_output_{}.npz".format(alpha_mode)))["alpha"]

    fig = plt.figure(figsize=(4.8, 3.75))
    # fig = plt.figure(figsize=(3.2, 2.5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xs, ys, c="r", s=5, label=r"$\alpha$")
    
    if mode == "AIC" or mode == "BIC":
        AICs = []
        for dim in range(1, 21):
            print("N=", str(dim), end=" ")

            error, coe = LSM(np.array([xs_unit, ys]).T, dim+1)
            # print("error:", error)
            # print("coe:", coe[::-1])

            l = l_MAX(list(coe[::-1]), list(xs), list(ys))
            if mode == "AIC":
                AIC_n = AIC(l, dim+1)
                print("AIC:", AIC_n)
            elif mode == "BIC":
                AIC_n = BIC(l, dim+1, sim_time)
                print("BIC:", AIC_n)
            AICs.append(AIC_n)

            best_dim = np.argmin(AICs)
            _, coe = LSM(np.array([xs_unit, ys]).T, best_dim+1)
            np.savez(os.path.join(work_dir, "coe.npz"), dim=best_dim+1, coe=coe)
            print("best dim:", best_dim)

            ax1.plot(xs, quation_LSM(coe, xs_unit), c="b")
    elif mode=="gauss":
        best_conditions = gauss_LSM(xs_unit, ys)
        amp = best_conditions["amp"]
        mu = best_conditions["mu"]
        var = best_conditions["var"]
        offset = best_conditions["offset"]
        approximate_y = gauss_func(xs_unit, amp, mu, var, offset)
        np.savez(os.path.join(work_dir, "gauss_approximate.npz"), y=approximate_y)

        ax1.plot(xs, approximate_y, c="b")
    elif mode=="gp":
        indexes = np.arange(xs.shape[0])
        rand_indexed = np.random.choice(indexes, 2700)
        rand_x = xs_unit[rand_indexed]
        print(rand_x.shape)
        rand_y = ys[rand_indexed]
        # kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        # kernel = RBF() + ConstantKernel()
        # kernel = ExpSineSquared(2) + ConstantKernel(2) + RBF()
        kernel = RBF()
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(rand_x.reshape(-1,1), rand_y)
        y_mu, y_sigma = model.predict(xs_unit.reshape(-1,1), return_std=True)
        print(y_mu.min())
        np.savez(os.path.join(work_dir, "gaussian_process_approximate.npz"), y=y_mu)
        
        ax1.plot(xs, y_mu, c="b")
    elif "linear":
        interpolate_index = np.concatenate([np.arange(0, 500, 10), np.arange(500, 3000, 100)])
        interpolate_index = np.append(interpolate_index, 2999)
        print(interpolate_index.shape)
        
        interpolate_f = interpolate.interp1d(interpolate_index, ys[interpolate_index], kind='linear')
        interpolated = interpolate_f(xs)
        
        np.savez(os.path.join(work_dir, "division_linear_alpha_{}.npz".format(alpha_mode)), y=interpolated)
        ax1.plot(xs, interpolated, c="b", label="approximation curve")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel(r"$\alpha$")
        ax1.legend()
        fig.tight_layout()
    else:
        raise ValueError

    fig.savefig(os.path.join(work_dir, "time_alpha_with_curve_{}.svg".format(alpha_mode)))
    
def plot_both_alpha(work_dir, sim_time):
    plt.style.use("ggplot")
    grid_y = np.load(os.path.join(work_dir, "alpha_output.npz"))["alpha"]
    grid_app_y = np.load(os.path.join(work_dir, "division_linear_alpha.npz"))["y"]
    emp_y = np.load(os.path.join(work_dir, "alpha_output_emp.npz"))["alpha"]
    emp_app_y = np.load(os.path.join(work_dir, "division_linear_alpha_emp.npz"))["y"]
    x = np.arange(sim_time)
    
    fig = plt.figure(figsize=(4.8, 3.75))
    # fig = plt.figure(figsize=(3.2, 2.5))
    ax1 = fig.add_subplot(111)
    
    ax1.scatter(x, grid_y, s=10, color=cm.tab10(3), alpha=0.1)
    grid_p, = ax1.plot(x, grid_app_y, linewidth=1.0, color=cm.tab10(3))
    grid_s, = ax1.plot([], [], marker="o", label="(Grid)", color=cm.tab10(3), alpha=0.2)
    
    
    ax1.scatter(x, emp_y, s=10, color=cm.tab10(0), alpha=0.1)
    emp_p, = ax1.plot(x, emp_app_y, linewidth=1.0, color=cm.tab10(0))
    emp_s, = ax1.plot([], [], marker="o", label="(Emp)", color=cm.tab10(0), alpha=0.2)
    
    ax1.set_xlabel("Time Step")
    ax1.set_xlim((-20, 1000))
    ax1.set_ylabel(r"$\alpha$")
    ax1.legend([(grid_s, grid_p), (emp_s, emp_p)], ['(Grid)', '(Emp)'])
    fig.tight_layout()
    
    fig.savefig(os.path.join(work_dir, "time_both_alpha.svg"))


if __name__ == "__main__":
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    work_dir = os.path.join(os.environ["DATADIR"], "models", "sdn-{}_{}".format(args.model, args.dataset), "{}".format(args.ic_index))
    # os.makedirs(work_dir, exist_ok=True)

    dataset_dir = os.path.join(os.environ["DATADIR"], "datasets")

    # spike_simulation(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir, sim_time=args.sim_time)
    # plot_alpha(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir, sim_time=args.sim_time, mode="exp")
    # plot_alpha_estimation(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir, sim_time=args.sim_time, mode=args.alpha_mode)
    # get_approximate_curve(ic_index=args.ic_index, work_dir=work_dir, dataset_dir=dataset_dir, sim_time=args.sim_time, alpha_mode=args.alpha_mode, mode="linear")
    plot_both_alpha(work_dir=work_dir, sim_time=args.sim_time)
