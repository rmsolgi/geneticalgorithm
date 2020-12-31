
import os
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
            
            plt.plot(report_list, color = main_color, label = 'best of generation', linewidth = 2)
            
            plt.xlabel('Generation')
            plt.ylabel('Minimized function')
            plt.title('GA optimization process')
            plt.legend()
            
            plt.savefig(os.path.join(folder, f"{file_prefix}_{generation_number}.png"), dpi = 200)

            if show: plt.show()
            else: plt.close()
        
        return func

        








