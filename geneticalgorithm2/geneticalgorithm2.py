'''

Copyright 2020 Ryan (Mohammad) Solgi, Demetry Pascal

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################

import sys
import time
import random


import numpy as np
from joblib import Parallel, delayed

from func_timeout import func_timeout, FunctionTimedOut

import matplotlib.pyplot as plt

###############################################################################

from .crossovers import Crossover
from .mutations import Mutations
from .selections import Selection
from .initializer import Population_initializer


###############################################################################

def can_be_prob(value):
    return value >=0 and value <=1

def is_numpy(arg):
    return type(arg) == np.ndarray

###############################################################################

class geneticalgorithm2():
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                report: a list including the record of the progress of the
                algorithm over iterations

    '''
    
    default_params = {
        'max_num_iteration': None,
        'population_size':100,
        'mutation_probability':0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type':'uniform',
        'mutation_type': 'uniform_by_center',
        'selection_type': 'roulette',
        'max_iteration_without_improv':None
        }
    
    
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters = default_params):
        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function. 
        (For maximization multiply function by a negative sign: the absolute 
        value of the output would be the actual objective function)
        
        @param dimension <integer> - the number of decision variables
        
        @param variable_type <string> - 'bool' if all variables are Boolean; 
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)
        
        @param variable_boundaries <numpy array/None> - Default None; leave it 
        None if variable_type is 'bool'; otherwise provide an array of tuples 
        of length two as boundaries for each variable; 
        the length of the array must be equal dimension. For example, 
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first 
        and upper boundary 200 for second variable where dimension is 2.
        
        @param variable_type_mixed <numpy array/None> - Default None; leave it 
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first 
        variable is integer but the second one is real the input is: 
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1] 
        in variable_boundaries. Also if variable_type_mixed is applied, 
        variable_boundaries has to be defined.
        
        @param function_timeout <float> - if the given function does not provide 
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function. 
        
        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int> 
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string/function> - Default is 'uniform'; 'one_point' or 'two_point' (not only) are other options
            @ mutation_type <string/function> - Default is 'uniform_by_x'; see GitHub to check other options
            @ selection_type <string/function> - Default is 'roulette'; see GitHub to check other options
            @ max_iteration_without_improv <int> - maximum number of successive iterations without improvement. If None it is ineffective
        
        for more details and examples of implementation please visit:
            https://github.com/PasaOpasen/geneticalgorithm2
  
        '''
        self.__name__ = geneticalgorithm2

        # input algorithm's parameters
        
        self.param = algorithm_parameters
        self.param.update({key:val for key, val in geneticalgorithm2.default_params.items() if key not in list((algorithm_parameters.keys()))})
        

        #############################################################
        # input function
        assert (callable(function)), "function must be callable!"     
        
        self.f = function
        #############################################################
        #dimension
        
        self.dim = int(dimension)
        
        #############################################################
        # input variable type
        
        assert(variable_type in ['bool', 'int', 'real']),  f"\n variable_type must be 'bool', 'int', or 'real', got {variable_type}"
       #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:
            
            if variable_type == 'real': 
                self.var_type = np.array([['real']]*self.dim)
            else:
                self.var_type = np.array([['int']]*self.dim)            

        else:
            assert (is_numpy(variable_type_mixed)), "\n variable_type must be numpy array"  
            assert (len(variable_type_mixed) == self.dim),  "\n variable_type must have a length equal dimension."       

            for i in variable_type_mixed:
                assert (i == 'real' or i == 'int'), "\n variable_type_mixed is either 'int' or 'real' "+"ex:['int','real','real']"+"\n for 'boolean' use 'int' and specify boundary as [0,1]"
                
            self.var_type = variable_type_mixed
            
        self.set_crossover_and_mutations(algorithm_parameters['crossover_type'], algorithm_parameters['mutation_type'], algorithm_parameters['selection_type'])
        #############################################################
        # input variables' boundaries 

            
        if variable_type != 'bool' or type(variable_type_mixed).__module__=='numpy':
                       
            assert (type(variable_boundaries).__module__=='numpy'),  "\n variable_boundaries must be numpy array"
        
            assert (len(variable_boundaries)==self.dim),\
            "\n variable_boundaries must have a length equal dimension"        
        
        
            for i in variable_boundaries:
                assert (len(i) == 2),  "\n boundary for each variable must be a tuple of length two." 
                assert(i[0]<=i[1]), "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0,1]]*self.dim)
 
        ############################################################# 
        #Timeout
        self.funtimeout = float(function_timeout)
        
        ############################################################# 

        
        self.pop_s = int(self.param['population_size'])
        
        assert ( can_be_prob( elf.param['parents_portion'] ) ), "parents_portion must be in range [0,1]" 
        
        self.par_s = int(self.param['parents_portion']*self.pop_s)
        trl= self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s+=1
               
        self.prob_mut = self.param['mutation_probability']
        
        assert (can_be_prob( self.prob_mut)), "mutation_probability must be in range [0,1]"
        
        
        self.prob_cross=self.param['crossover_probability']
        assert (can_be_prob( self.prob_cross)),  "mutation_probability must be in range [0,1]"
        
        assert ( can_be_prob( self.param['elit_ratio']) ),  "elit_ratio must be in range [0,1]"                
        
        trl = self.pop_s*self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)
            
        assert(self.par_s>=self.num_elit), "\n number of parents must be greater than number of elits"
        
        if self.param['max_num_iteration'] == None:
            self.iterate = 0
            for i in range (0, self.dim):
                if self.var_type[i]=='int':
                    self.iterate += (self.var_bound[i][1]-self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1]-self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate*self.pop_s)>10000000:
                self.iterate=10000000/self.pop_s
            
            if self.iterate > 8000:
                self.iterate = 8000

        else:
            self.iterate = int(self.param['max_num_iteration'])
        
        
        self.stop_mniwi=False ## what
        if self.param['max_iteration_without_improv'] == None or self.param['max_iteration_without_improv'] < 1:
            self.mniwi = self.iterate+1
        else: 
            self.mniwi = int(self.param['max_iteration_without_improv'])

        

    def set_crossover_and_mutations(self, crossover, mutation, selection):
        
        if type(crossover) == str:
            if crossover == 'one_point':
                self.crossover = Crossover.one_point()
            elif crossover == 'two_point':
                self.crossover = Crossover.two_point()
            elif crossover == 'uniform':
                self.crossover = Crossover.uniform()
            elif crossover == 'segment':
                self.crossover = Crossover.segment()
            elif crossover == 'shuffle':
                self.crossover = Crossover.shuffle()
            else:
                raise Exception(f"unknown type of crossover: {crossover}")
        else:
            self.crossover = crossover 
        
        if type(mutation) == str:
            if mutation == 'uniform_by_x':
                self.real_mutation = Mutations.uniform_by_x()
            elif mutation == 'uniform_by_center':
                self.real_mutation = Mutations.uniform_by_center()
            elif mutation == 'gauss_by_center':
                self.real_mutation = Mutations.gauss_by_center()
            elif mutation == 'gauss_by_x':
                self.real_mutation = Mutations.gauss_by_x()
            else:
                raise Exception(f"unknown type of mutation: {mutation}")
        else:
            self.real_mutation = mutation
            
        if type(selection) == str:
            if selection == 'fully_random':
                self.selection = Selection.fully_random()
            elif selection == 'roulette':
                self.selection = Selection.roulette()
            elif selection == 'stochastic':
                self.selection = Selection.stochastic()
            elif selection == 'sigma_scaling':
                self.selection = Selection.sigma_scaling()
            elif selection == 'ranking':
                self.selection = Selection.ranking()
            elif selection == 'linear_ranking':
                self.selection = Selection.linear_ranking()
            elif selection == 'tournament':
                self.selection = Selection.tournament()
            else:
                raise Exception(f"unknown type of selection: {selection}")                
        else:
            self.selection = selection


        ############################################################# 
    def run(self, no_plot = False, \
            disable_progress_bar = False, \
            set_function = None, \
            apply_function_to_parents = False, \
            start_generation = {'variables':None, 'scores': None}, \
            studEA = False, \
            population_initializer = Population_initializer(select_best_of = 1, local_optimization_step = 'never', local_optimizer = None), \
            seed = None):
        """
        @param no_plot <boolean> - do not plot results using matplotlib by default
        
        @param disable_progress_bar <boolean> - do not show progress bar
                
        @param set_function : 2D-array -> 1D-array function, which applyes to matrix of population (size (samples, dimention))
        to estimate their values
        
        @param apply_function_to_parents <boolean> - apply function to parents from previous generation (if it's needed)
                                                                                                         
        @param start_generation <dictionary> - a dictionary with structure {'variables':2D-array of samples, 'scores': function values on samples}                                                                                                
        if 'scores' value is None the scores will be compute

        @param studEA <boolean> - using stud EA strategy (crossover with best object always)

        @param population_initializer (tuple(int, func)) - object for actions at population initialization step to create better start population. See doc

        @ param seed - random seed (None is doesn't matter)
        """
        
        if not (seed is None):
            random.seed(seed)
            np.random.seed(seed)

        show_progress = (lambda t, t2, s: self.progress(t, t2, status = s)) if not disable_progress_bar else (lambda t, t2, s: None)
        
        get_parents_inds = (lambda par_count: (0, random.randrange(1, par_count))) if studEA else (lambda par_count: tuple(np.random.randint(0, par_count, 2)))
            
        ############################################################# 
        # Initial Population
        
        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')
        
        if set_function == None:
            set_function = geneticalgorithm2.default_set_function(self.f)
        
        pop_coef, initializer_func = population_initializer
        
        
        if start_generation['variables'] is None:
                    
            pop = np.empty((self.pop_s*pop_coef, self.dim+1)) #np.array([np.zeros(self.dim+1)]*self.pop_s)
            solo = np.empty(self.dim+1)
            var = np.empty(self.dim)       
            
            for i in self.integers[0]:
                pop[:, i] = np.random.randint(self.var_bound[i][0],self.var_bound[i][1]+1, self.pop_s*pop_coef) 
            
            for i in self.reals[0]:
                pop[:, i] = np.random.uniform(self.var_bound[i][0], self.var_bound[i][1], self.pop_s*pop_coef)
            
            for p in range(0, self.pop_s*pop_coef):    
                var = pop[p, :-1]
                solo = pop[p, :]
                obj = self.sim(var)  # simulation returns exception or func value           
                solo[self.dim] = obj
                
        else:
            assert (start_generation['variables'].shape[1] == self.dim), f"incorrect dimention ({start_generation['variables'].shape[1]}) of population, should be {self.dim}"
            
            self.pop_s = start_generation['variables'].shape[0]
            pop = np.empty((self.pop_s, self.dim+1))
            pop[:,:-1] = start_generation['variables']
            
            if not (start_generation['scores'] is None):
                assert (start_generation['scores'].size == start_generation['variables'].shape[0]), f"count of samples ({start_generation['variables'].shape[0]}) must be equal score length ({start_generation['scores'].size})"
                
                pop[:, -1] = start_generation['scores']
            else:
                pop[:, -1] = set_function(pop[:,:-1])
            
            obj = pop[-1, -1] # some evaluated value
            var = pop[-1, :-1] # some variable
            solo = pop[-1, :]
        
        
        # Initialization by select bests and local_descent
        
        population, scores = initializer_func(pop[:, :-1], pop[:,-1])
        
        pop = np.hstack([population, scores[:, np.newaxis]])
        self.pop_s = pop.shape[0]
        
        #############################################################

        #############################################################
        # Report
        self.report = []
        self.report_min = []
        self.report_average = []
        
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj
        ##############################################################   
        
        t = 1
        counter = 0
        while t <= self.iterate:
            
            
            show_progress(t, self.iterate, f"GA is running...{t} gen from {self.iterate}")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

                
            if pop[0,self.dim] < self.best_function: # if there is progress
                counter = 0
                self.best_function=pop[0, self.dim]
                self.best_variable=pop[0,:self.dim].copy()
                
                show_progress(t, self.iterate, f"GA is running...{t} gen from {self.iterate}...best value = {self.best_function}")
            else:
                counter += 1
            #############################################################
            # Report

            self.report.append(pop[0, self.dim])
            self.report_min.append(pop[-1, self.dim])
            self.report_average.append(np.mean(pop[:, self.dim]))
    

  
            #############################################################        
            # Select parents
            
            par = np.empty((self.par_s, self.dim+1))
            
            # elit parents
            par[:self.num_elit, :] = pop[:self.num_elit, :].copy()
                
            # non-elit parents indexes
            #new_par_inds = np.int8(ff(pop[self.num_elit:, self.dim], self.par_s - self.num_elit) + self.num_elit)
            new_par_inds = np.int8(self.selection(pop[:, self.dim], self.par_s - self.num_elit))
            par[self.num_elit:self.par_s] = pop[new_par_inds].copy()
            
            
            # select parents for crossover
            par_count = 0
            while par_count < 2:                
                ef_par_list = np.random.random(self.par_s) <= self.prob_cross
                par_count = np.sum(ef_par_list)
                 
            ef_par = par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop = np.empty((self.pop_s, self.dim+1))  #np.array([np.zeros(self.dim+1)]*self.pop_s)
            
            pop[:self.par_s, :] = par[:self.par_s, :].copy()
                
            for k in range(self.par_s, self.pop_s, 2):
                r1, r2 = get_parents_inds(par_count)
                pvar1 = ef_par[r1,:self.dim].copy()
                pvar2 = ef_par[r2,:self.dim].copy()
                
                ch1, ch2 = self.crossover(pvar1, pvar2)
                
                if self.prob_mut > 0:
                    ch1 = self.mut(ch1)
                    ch2 = self.mutmidle(ch2, pvar1, pvar2)               
                
                solo[: self.dim] = ch1.copy()                
                #solo[self.dim] = self.sim(ch1)
                pop[k] = solo.copy()                
                
                solo[: self.dim] = ch2.copy()                             
                #solo[self.dim] = self.sim(ch2) 
                pop[k+1] = solo.copy()
                
            
            if apply_function_to_parents:
                pop[:,-1] = set_function(pop[:,:-1])
            else:
                pop[self.par_s:,-1] = set_function(pop[self.par_s:,:-1])
            
            
            
            
        #############################################################       
            t += 1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim] >= self.best_function:
                    t = self.iterate # to stop loop
                    show_progress(t, self.iterate, "GA is running... STOP!")
                    #time.sleep(0.7) #time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
        
        
        
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim] < self.best_function:
                
            self.best_function=pop[0,self.dim]#.copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0,self.dim])
        self.report_min.append(pop[-1, self.dim])
        self.report_average.append(np.mean(pop[:, self.dim]))
        
        
 
        self.output_dict = {
            'variable': self.best_variable, 
            'function': self.best_function,
            'last_generation': {
                'variables':pop[:, :-1],
                'scores': pop[:, -1]
                }
            }
        
        show=' '*200
        sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush() 
        
        if not no_plot:
            self.plot_results()

        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')
##############################################################################         

    def plot_results(self, show_mean = False):
        """
        Simple plot of self.report (if not empty)
        """
        if len(self.report) == 0:
            sys.stdout.write("No results to plot!\n")
            return

        bests = np.array(self.report)
        means = np.array(self.report_average)
        
        if show_mean: plt.plot(means, color = 'red', label = 'mean by generation', linewidth = 1)
        plt.plot(bests, color = 'blue', label = 'best of generation')
        
        plt.xlabel('Generation')
        plt.ylabel('Minimized function')
        plt.title('Genetic Algorithm')
        plt.legend()
        plt.show()



###############################################################################  
    
    def mut(self,x):
        """
        just mutation
        """
        
        for i in self.integers[0]:
            if np.random.random() < self.prob_mut:
                x[i]=np.random.randint(self.var_bound[i][0], self.var_bound[i][1]+1) 
                    
        

        for i in self.reals[0]:                
            if np.random.random() < self.prob_mut:
                x[i] = self.real_mutation(x[i], self.var_bound[i][0], self.var_bound[i][1]) 
            
        return x

###############################################################################
    def mutmidle(self, x, p1, p2):
        """
        mutation oriented on parents
        """
        
        for i in self.integers[0]:

            if np.random.random() < self.prob_mut:
                if p1[i]<p2[i]:
                    x[i]=np.random.randint(p1[i],p2[i])
                elif p1[i]>p2[i]:
                    x[i]=np.random.randint(p2[i],p1[i])
                else:
                    x[i]=np.random.randint(self.var_bound[i][0], self.var_bound[i][1]+1)
                        
        for i in self.reals[0]:                
            ran=np.random.random()
            if ran < self.prob_mut:   
                if p1[i] != p2[i]:
                    x[i]=np.random.uniform(p1[i], p2[i]) 
                else:
                    x[i]= np.random.uniform(self.var_bound[i][0], self.var_bound[i][1])
        return x


###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
    
###############################################################################    
    def sim(self, X):
        
        self.temp = X#.copy()
        
        obj = None
        try:
            obj = func_timeout(self.funtimeout, self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        
        assert (obj is not None), "After "+str(self.funtimeout)+" seconds delay "+\
                "func_timeout: the given function does not provide any output"
                
        assert ((type(obj)==int or type(obj)==float or obj.size==1)), "Function should return a number or an np.array with len == 1"
        
        return obj

###############################################################################
    def progress(self, count, total, status=''):
        
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()     
        
###############################################################################
    @staticmethod
    def default_set_function(function_for_set):
        """
        simple function for creating set_function 
        function_for_set just applyes to each row of population
        """
        def func(matrix):
            return np.array([function_for_set(matrix[i,:]) for i in range(matrix.shape[0])])
        return func
    
    def set_function_multiprocess(function_for_set, n_jobs = -1):
        """
        like function_for_set but uses joblib with n_jobs (-1 goes to count of available processors)
        """
        def func(matrix):
            result = Parallel(n_jobs=n_jobs)(delayed(function_for_set)(matrix[i,:]) for i in range(matrix.shape[0]))
            return np.array(result)
        return func
            
###############################################################################



















            
            