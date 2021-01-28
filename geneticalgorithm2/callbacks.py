
import os
import random

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def folder_create(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class Callbacks:
    
    @staticmethod
    def NoneCallback():
        return lambda generation_number, report_list, last_population, last_scores: None


    @staticmethod
    def SavePopulation(folder, save_gen_step = 50, file_prefix = 'population'):
        
        folder_create(folder)

        def func(generation_number, report_list, last_population, last_scores):
    
            if generation_number % save_gen_step != 0:
                return

            np.savez(os.path.join(folder, f"{file_prefix}_{generation_number}.npz"), population=last_population, scores=last_scores)
        
        return func
    
    @staticmethod
    def PlotOptimizationProcess(folder, save_gen_step = 50, show = False, main_color = 'green', file_prefix = 'report'):
        folder_create(folder)

        def func(generation_number, report_list, last_population, last_scores):

            if generation_number % save_gen_step != 0:
                return

            # if len(report_list) == 0:
            #     sys.stdout.write("No results to plot!\n")
            #     return
            
            ax = plt.axes()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            plt.plot(np.arange(1, 1 + len(report_list)), report_list, color = main_color, label = 'best of generation', linewidth = 2)
            
            plt.xlabel('Generation')
            plt.ylabel('Minimized function')
            plt.title('GA optimization process')
            plt.legend()
            
            plt.savefig(os.path.join(folder, f"{file_prefix}_{generation_number}.png"), dpi = 200)

            if show: plt.show()
            else: plt.close()
        
        return func






class Actions:

    @staticmethod
    def Stop():
        
        def func(data):
            data['current_stagnation'] = 2*data['max_stagnation']
            return data
        return func

    @staticmethod
    def ReduceMutationProb(reduce_coef = 0.9):
        
        def func(data):
            data['mutation_prob'] *= reduce_coef
            return data
        
        return func


    #def DualStrategyStep():
    #    pass

    #def SetFunction():
    #    pass



    @staticmethod
    def ChangeRandomCrossover(available_crossovers):

        def func(data):
            data['crossover'] = random.choice(available_crossovers)
            return data

        return func
    
    @staticmethod
    def ChangeRandomSelection(available_selections):

        def func(data):
            data['selection'] = random.choice(available_selections)
            return data

        return func

    @staticmethod
    def ChangeRandomMutation(available_mutations):

        def func(data):
            data['mutation'] = random.choice(available_mutations)
            return data

        return func



class ActionConditions:
    
    @staticmethod
    def EachGen(generation_step = 10):
        def func(data):
            return data['current_generation'] % generation_step == 0 and data['current_generation'] > 0
        return func

    @staticmethod
    def AfterStagnation(stagnation_generations = 50):

        def func(data):
            return data['current_stagnation'] % stagnation_generations == 0 and data['current_stagnation'] > 0
        return func


    @staticmethod
    def Several(list_of_conditions):
        """
        returns function which checks all conditions from list_of_conditions
        """

        def func(data):
            return all((cond(data) for cond in list_of_conditions))
        
        return func




class MiddleCallbacks:

    @staticmethod
    def UniversalCallback(action, condition):
        
        def func(data):

            cond = condition(data)
            if cond:
                data = action(data)

            return data, cond
        
        return func

    #def ReduceMutationGen(reduce = 0.9, each_generation = 50):
    #    pass

    #def ReduceMutationStagnation(reduce = 0.5, stagnation_gens = 50):
    #    pass 
       








