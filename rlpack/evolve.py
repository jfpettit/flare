import torch
import torch.distributions as tdist
import evograd
from evograd.distributions import Normal
import evograd.expectation as expectation
import sys

class ES:
    def __init__(self, env, population_size=100, mean=1.0, std=0.5, alpha=3e-2):
        self.popsize = population_size
        self.mu = torch.tensor([mean], requires_grad=True)
        self.std = std
        self.alpha = alpha
        self.distribution = Normal(self.mu, self.std)
        self.env = env

    def makepop(self, net, wstd=0.01):
        population = []
        mus = self.distribution.sample(self.popsize)
        for mu in mus:
            enet = net(self.env.observation_space.shape[0], self.env.action_space.n)
            net_state_dict = enet.state_dict()
            keylist = list(net_state_dict.keys())
            for key in (keylist):
                l_count = torch.tensor([sum(param.numel() for param in net_state_dict[key])])
                l_shape = net_state_dict[key].shape
                net_state_dict[key] = tdist.Normal(mu, wstd).sample(l_count).reshape(l_shape) #* torch.sqrt(2/l_count.float())
            enet.load_state_dict(net_state_dict)
            population.append(enet)
        return population, mus


    def playenv(self, population, EPOCHS=5):
        returns = []
        for member in population:
            epret = 0
            for epoch in range(EPOCHS):
                obs, done = self.env.reset(), False
                while not done:
                    action = tdist.Categorical(member(torch.from_numpy(obs).float())).sample()
                    obs, r, done, _ = self.env.step(action.numpy())
                    epret += r
            returns.append(epret/EPOCHS)
        return returns

    def backprop_mean(self, fitnesses, popmeans, mu):
        s = torch.tensor(popmeans) if type(popmeans) is not torch.Tensor else popmeans
        mean_fit = expectation(fitnesses, s, p=self.distribution)
        mean_fit.backward()
        with torch.no_grad():
            mu += self.alpha * mu.grad
            mu.grad.zero_()
        return mu

    def evolution(self, net, wstd=0.01, generations=50, EPOCHS=5, standardize_fits=False, anneal_std=False, anneal_wstd=False, solved_threshold=None):
        for generation in range(generations):
            if anneal_std:
                stdrange = torch.linspace(self.std, self.std/generations, steps=generations)
                self.std = stdrange[generation]
            if anneal_wstd:
                wstdrange = torch.linspace(wstd, wstd/generations, steps=generations)
                wstd = wstdrange[generation]
            pop, popm = self.makepop(net, wstd=wstd)
            self.fits = torch.tensor(self.playenv(pop, EPOCHS=EPOCHS))
            fitsmean = self.fits.mean()
            maxfit = torch.max(self.fits)
            if solved_threshold is not None:
                if maxfit >= solved_threshold:
                    print('\r Early stopping. Top performer has {} fitness, above {} threshold'.format(maxfit, solved_threshold))
                    break
            if standardize_fits:
                self.fits = (self.fits - self.fits.mean()) / self.fits.std()
            self.mu = self.backprop_mean(self.fits, popm, self.mu)
            self.population = pop
            print('\r Generation {} average fitness {}. Best performer {}'.format(generation, fitsmean, maxfit), end="")
            sys.stdout.flush()
        print('\n')

    def get_best_state_dict(self):
        return self.population[torch.argmax(self.fits)].state_dict()

