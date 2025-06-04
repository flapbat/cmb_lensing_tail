import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import copy

import scipy
import scipy.linalg
from scipy import integrate

class candFunction():
    
    # Candidate Function
    
    def __init__(self):
        
        self.tot_genes = 12 + 2 + 7 + 1
        # 12 wayne damping parameters..., 
        # 2 -> l_d & m, 
        # 7 -> simple polynomial expression for alpha
        # 1 -> offset
        
        self.genes     = np.zeros(self.tot_genes)
        
        self.w_genes = [ 2.25719934e-02,  3.57192439e-02,  1.62856750e+01,  3.18532412e-01, 3.73023816e+02, -1.92326169e-01, -7.01447041e+00, -2.98206510e-01, 1.13259556e+00,  3.26604853e-02,  3.89733328e-02,  9.92579509e-02]
        
        # ignore this next line.... it's here 
        self.gene_ranges = np.array([[-10, 10] for i in range(self.tot_genes)])
        
        self.creation_time = datetime.datetime.now()
    
    def age_reset(self):
        self.creation_time = datetime.datetime.now()
    
    def set_genes(self, genes):
        self.genes = genes
        
    def set_ranges(self, gene_ranges):
        self.gene_ranges = gene_ranges
    
    def randomize_genes(self):
        self.genes = np.random.uniform(self.gene_ranges[:, 0], self.gene_ranges[:, 1])
        
    def mutate(self, T, ngenes_to_mutate: int = 1):
        
        # select genes to mutate
        inds = [i for i in range(self.tot_genes)]
        mutate_inds = np.random.choice(inds, ngenes_to_mutate, replace = False)

        for i in mutate_inds:
            
            factor = (1.1 + T)**(2 * np.random.rand() - 1)
        
            if np.random.rand() < 0.4: # CHANGED TO ZERO FOR TESTING PURPOSES
                factor *= -1
            
            self.genes[i] *= factor

    # COMPONENTS OF THE FUNCTION vvv
    
    # If you want to change the functional form... make sure you put self.genes[X] wherever GA shoud learn a value
    # then make sure self.tot_genes is equal to the total number of genes
    # and finally, make sure n_genes in the testing notebook is equal to self.tot_genes
            
    def par_to_a(self, par):
        
        omega_0 = par[0] + par[1]
        
        a1 = self.genes[0]*omega_0**(self.genes[1])*(1+self.genes[2]*omega_0**(self.genes[3]))
        a2 = self.genes[4]*omega_0**(self.genes[5])/(1+self.genes[6]*omega_0**(self.genes[7]))
        a3 = self.genes[8]*omega_0**(self.genes[9])
        a4 = self.genes[10]*omega_0**(self.genes[11])
        
        return np.array([a1, a2, a3, a4])
    
    def par_to_r_theta(self, par):
        omega_b=par[0]
        omega_0=par[0]+par[1]
        c=3e5
        H0=par[2]
        h=H0/100
        a_eq=(4.17e-5)/omega_0*(2.726/2.728)**4
        b1=0.0783*omega_b**(-0.238)/(1+39.5*omega_b**(0.763))
        b2=0.56/(1+21.1*omega_b**(1.81))
        a_star=1/(1+1048*(1+0.00124*omega_b**(-0.738))*(1+b1*(omega_0**b2)))
        integrand=lambda x:1/H0/np.sqrt(omega_0/h**2*x**(-3)+(1-omega_0/h**2)+2.47e-5/h**2*x**(-4))/x**2
        eta_0=integrate.quad(integrand, 0,1)[0]#2*(omega_0*10000)**(-0.5)*(np.sqrt(1+a_eq)-np.sqrt(a_eq))*(1-0.0841*np.log(omega_0/h**2))
        eta_star=integrate.quad(integrand,0, a_star)[0]#2*(omega_0*10000)**(-0.5)*(np.sqrt(a_star+a_eq)-np.sqrt(a_eq))
        return (eta_0-eta_star)*c
    
    def par_to_lD_m(self, par):
        omega_b=par[0]
        omega_0=par[0]+par[1]
        a=self.par_to_a(par) # CAND_FUNCTION
        r_theta=self.par_to_r_theta(par)
        l_D=r_theta*a[0]*omega_b**(0.291)*(1+a[1]*omega_b**(1.8))**(-0.2)
        m=a[2]*omega_b**(a[3])*(1+omega_b**(1.8))**(0.2)
        return np.array([l_D,m])

    def par_to_D_l(self, ell, par):
        l_D,m=self.par_to_lD_m(par)
        return np.exp(-(ell/l_D)**m)


    def par_to_lr(self, par):
        
        omega_b=par[0]
        omega_0=par[0]+par[1]
        #c=3e5
        H0=par[2]
        h=H0/100
        a_eq=(4.17e-5)/omega_0*(2.726/2.728)**4

        eta_0=2*(omega_0*10000)**(-0.5)*(np.sqrt(1+a_eq)-np.sqrt(a_eq))*(1-0.0841*np.log(omega_0/h**2))
        eta_r=2*(omega_0*10000)**(-0.5)*(np.sqrt(1/11+a_eq)-np.sqrt(a_eq)) #picked z=10 for reionizatoin

        return (eta_0-eta_r)/eta_r


    def par_to_Rl2(self, ell, par):
        tau=par[3]
        lr=self.par_to_lr(par)
        x=ell/(lr+1)
        c1=-0.276
        c2= 0.581
        c3=-0.172
        c4= 0.0312
        return np.exp(-2*tau)+(1-np.exp(-2*tau))/(1+c1*x+c2*x**2+c3*x**3+c4*x**4)

    def par_to_P_l(self, ell, par):
        rho_nu=7/8*3.046*(4/11)**(4/3)
        rho_gamma=1
        f_nu=rho_nu/(rho_nu+rho_gamma)
        omega_b=par[0]
        omega_0=par[0]+par[1]
        b1=0.0783*omega_b**(-0.238)/(1+39.5*omega_b**(0.763))
        b2=0.56/(1+21.1*omega_b**(1.81))
        a_star=1/(1+1048*(1+0.00124*omega_b**(-0.738))*(1+b1*(omega_0**b2)))
        R_star=omega_b*30000*a_star
        omega_0=par[0]+par[1]
        c=3e5
        H0=par[2]
        h=H0/100
        r_theta=self.par_to_r_theta(par)
        a_eq=(4.17e-5)/omega_0
        k_eq=np.sqrt(omega_0*10000*2/a_eq)
        l_eq=r_theta*k_eq/c
        A=25/(1+4/15*f_nu)**2*(1/np.sqrt(1+R_star)+(1+R_star)**(-3/2))/2-1
        #print(l_eq)
        return A*np.exp(-1.4*l_eq/ell)+1

    def par_to_rescale(self, par, camb_ell_min = 2, camb_ell_max = 5000):
        ell=np.arange(camb_ell_min,camb_ell_max,1)
        A_s=np.exp(par[5])
        tau=par[3]
        rescale=A_s/np.exp(2*tau)*(self.par_to_D_l(ell,par))**2*self.par_to_Rl2(ell,par)*self.par_to_P_l(ell,par)
        return rescale
    
    def par_to_alpha(self, par):
        
        # c_1*(ombh2)**c_2 + c_3*(omch2)**c_4 + c_5*(ombh2**c_6)*(omch2**c_7) - 3.3
        #                                                                       ^ guessed
        
        
        offset = self.genes[21]# -3.3  # note: I can make this learnable by setting equal to self.genes[x] ...
        
        val = self.genes[12]*(par[0]**self.genes[13]) + self.genes[14]*(par[1]**self.genes[15]) + self.genes[16]*(par[0]**self.genes[17])*(par[0]**self.genes[18])
        val += offset
        
        return val
    
    def par_to_lensing(self, par, camb_ell_min = 2, camb_ell_max = 5000):
        
        m     = self.genes[19] # initial guess: 1.2
        ell_d = self.genes[20] # initial guess: 1200 
        ell_c = 4000
        
        alpha = self.par_to_alpha(par)
        
        ell = np.arange(camb_ell_min,camb_ell_max,1)
        
        w_ell = (1+np.exp(-(ell-ell_c)/100))**(-1)
        
        val = 1 - w_ell
        
        val += w_ell * ((ell/ell_d)**alpha) / (np.exp(-(ell/ell_d)**m)) 
        
        return val
        
    
    def compute(self, par):
        val = self.par_to_rescale(par, 2, 5000) * self.par_to_lensing(par, 2, 5000)
        return val
    
# ------------------------------ GENETIC ALGORITHM CLASS ------------------------------

class geneticAlgorithm():
    
    # Fixed functional form, parameters learned
    
    def __init__(self, config, data_points):
        
        # compress all input data into a config dict
        self.n_genes = config["n_genes"]
        self.generations = config["generations"]
        self.population = config["population"]
        self.pop_inds = np.array([i for i in range(self.population)])
        self.ranges = config["ranges"]
        self.do_dynamic_ranges = config["do_dynamic_ranges"]
        self.init_genes = config["init_genes"]
        self.num_mutations = config["num_mutations"]
        self.tournament_size = config["tournament_size"]
        self.mutation_probs = config["mutation_probs"]
        self.batch_size = config["batch_size"]
        
        
        if type(self.ranges) is np.ndarray:
            self.set_ranges = True
        else:
            self.set_ranges = False
        
        if type(self.init_genes) is np.ndarray:
            self.set_genes = True
        else:
            self.set_genes = False
        
        
        self.kids = np.empty(self.population, dtype=candFunction)
        self.fitnesses = np.zeros(self.population)
        
        # store the best fitness measure per generation
        self.best_fit_per_gen = np.zeros(self.generations)
        
        self.best_fit_func = candFunction()
        
        self.data_points = data_points
        # list of dicts.... ["H0": 68, "ombh2": ......, "cls_lensed": [.....]]
        
        self.p_crossbreed = config["p_crossbreed"]
        
    
    def init_pop(self):
        
        for i in range(self.population):
            self.kids[i] = candFunction()
            
            if self.set_ranges:
                self.kids[i].set_ranges(self.ranges)
                
            if self.set_genes:
                
                # initialize the given initial_genes
                self.kids[i].set_genes(self.init_genes)
                
                # keep the first 80% of the population unchanged
                if i > 0.8*self.population:
                    
                    for k in range(self.n_genes//3):
                        self.kids[i].mutate(0.1)
            else:
                self.kids[i].randomize_genes()
                
            self.fitnesses[i] = self.compute_fitness(self.kids[i])
            
        print("Initial population generated.")
    
    def tournament(self):
        
        sub_population = np.random.choice(self.pop_inds, size = self.tournament_size)
        
        best_fit_sub_ind = np.nanargmin(self.fitnesses[sub_population])
        
        best_fit_ind = sub_population[best_fit_sub_ind]
        
        return best_fit_ind
    
    def update_ranges(self, gen):
        
        print("Updating ranges...")
        
        updated_ranges = np.zeros((self.n_genes, 2))
        
        where_finite = [i for i in range(self.population) if np.isfinite(self.fitnesses[i])]
        
        
        weights = np.zeros(self.population)
        
        weights[where_finite] = np.max(np.log10(self.fitnesses[where_finite])) - np.log10(self.fitnesses[where_finite])
        weights /= weights.sum()
        
       
        for i in range(self.n_genes):
            
            range_max = 0
            
            for j in range(self.population):
                range_max += (weights[j]*np.abs(self.kids[j].genes[i]))
                
            range_window_factor_size = np.exp(-2*(gen/self.generations)**4)
            
            updated_ranges[i, :] = [range_max*(1 - 9*range_window_factor_size/10), range_max*(1 + 9*range_window_factor_size/10)]
        
        self.ranges = updated_ranges
        
        self.update_kids_ranges()
        
        print("Ranges updated.")
    
    def update_kids_ranges(self):
        
        for i in range(self.population):
            self.kids[i].set_ranges(self.ranges)
    
    def run_algorithm(self):
        
        # generate random initial population
        self.init_pop()
        
        for gen in range(self.generations):
            if self.do_dynamic_ranges:
                self.update_ranges(gen)
            
            print(f"g:{gen + 1}. f:{np.log10(np.nanmin(self.fitnesses))}" )
            for i in range(self.num_mutations):
                
                if np.random.rand() > self.p_crossbreed:
                    if i % 10 == 0:
                        print(f"m:{i+1}/{self.num_mutations}")
                    func_to_mutate = copy.deepcopy(self.kids[self.tournament()])
                    func_to_mutate.age_reset()
                    
                    # possible mutations:
                        # mutate one gene
                        # mutate a couple genes
                        # completely new set of params
                    
                    action = np.random.choice(["mutate_single_gene", "mutate_multiple_genes", "new_expression", "do_nothing"], p=self.mutation_probs)
                
                    T = 1 - i/self.num_mutations # ANNEALING TEMPERATURE 

                    if action == "mutate_single_gene":
                        func_to_mutate.mutate(T) # take input T, use this in the random constant mutation

                    elif action == "mutate_multiple_genes":
                        n = np.random.randint(low = 0, high = self.n_genes)
                        func_to_mutate.mutate(T, ngenes_to_mutate = n)

                    elif action == "new_expression":
                        func_to_mutate.randomize_genes()
                        func_to_mutate.age_reset()
                        
                    elif action == "do_nothing":
                        continue
                    
                    new_fitness = self.compute_fitness(func_to_mutate)
                    
                    if not np.isnan(new_fitness):
                        oldest_ind = self.oldest_func_inds()[0]
                        self.kids[oldest_ind] = func_to_mutate
                        self.fitnesses[oldest_ind] = new_fitness

                else:
                    
                    # select two functions via tournament
                    
                    par1 = self.kids[self.tournament()]
                    par2 = self.kids[self.tournament()]
                    
                    kid1, kid2 = self.crossbreed(par1, par2)
                    
                    swap_ind1, swap_ind2 = self.oldest_func_inds()
                    
                    self.kids[swap_ind1] = kid1
                    self.fitnesses[swap_ind1] = self.compute_fitness(kid1)
                    
                    self.kids[swap_ind2] = kid2
                    self.fitnesses[swap_ind2] = self.compute_fitness(kid2)
                    
            # --------------------------------     
            # after all mutations are complete
            # store the best set of params(s)
        
        print("Evaluating final model fitnesses...\n(This may take some time.)")
        
        for i, kid in enumerate(self.kids):
            print(f"{i}/{self.population}")
            self.fitnesses[i] = self.compute_fitness(self.kids[i], mode = "all_data")

        self.best_fit_func = self.kids[np.nanargmin(self.fitnesses)]
        
    def oldest_func_inds(self):
        oldest_inds = [0, 1]
        
        for i in range(2, self.population):
            if self.kids[i].creation_time < self.kids[oldest_inds[0]].creation_time:
                oldest_inds[1] = oldest_inds[0]
                oldest_inds[0] = i
            elif self.kids[i].creation_time < self.kids[oldest_inds[1]].creation_time:
                oldest_inds[1] = i
                
        return oldest_inds[0], oldest_inds[1]
    
    
    def compute_fitness(self, func, mode = "batch"):
        
        if mode == "batch":
            batch = np.random.choice(self.data_points, self.batch_size)
        elif mode == "all_data":
            batch = self.data_points
        
        fitness = 0
        
        val_a = 0
        val_b = 0
        val_c = 0
        val_d = 0
        
        for dat in batch:
            
            par = [dat["ombh2"], dat["omch2"], dat["H0"], dat["tau"], 1, dat["As"]]
                 #[ombh2, omch2, H0, tau, n_s, As]
            damping_fit = func.par_to_rescale(par)
            lensing_fit = func.par_to_lensing(par)
            
            cutoff = 500
            
            a = 0#((dat["cls_lensed"]/(lensing_fit) - dat["cls_unlensed"])**2)[:cutoff].sum()
            b = ((dat["cls_lensed"]/(dat["cls_unlensed"]*lensing_fit) - 1)**2)[:cutoff].sum()*10e2
            c = ((dat["cls_lensed"]/lensing_fit/(damping_fit) - 390)**2).sum()/10e4
            d = ((dat["cls_lensed"]/lensing_fit/(damping_fit) - 390)**2)[3700:].sum()/(10e4)
               
            val_a += a
            val_b += b
            val_c += c
            val_d += d
        
        print(f"b: {np.log10(val_b)}, c: {np.log10(val_c)}, d: {np.log10(val_d)}")
        
        fitness += val_a + val_b + val_c + val_d
        
        # Eq. (16)
        fitness *= 1/len(batch)
        return fitness
    
    
    def crossbreed(self, parent1, parent2):
    
        par1_genes = parent1.genes
        par2_genes = parent2.genes
        
        kid1_genes = np.array([np.random.choice([par1_genes[i], par2_genes[i]]) for i in range(parent1.tot_genes)])
        kid2_genes = np.array([np.random.choice([par1_genes[i], par2_genes[i]]) for i in range(parent1.tot_genes)])
        # --------------------------------------------------

        kid1 = candFunction()
        kid2 = candFunction()
        kid1.set_genes(kid1_genes)
        kid2.set_genes(kid2_genes)
        
        if self.set_ranges:
            kid1.set_ranges(self.ranges)
            kid2.set_ranges(self.ranges)
            
        return kid1, kid2

